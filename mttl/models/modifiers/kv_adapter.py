from dataclasses import dataclass
import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mttl.models.modifiers import register_modifier
from functools import partial
from typing import Optional, Tuple
from mttl.utils import logger
from einops import rearrange, repeat
from mttl.models.modifiers.base import (
    Adapter,
    ModifierConfig,
    ModifyMixin,
    ModifierConfig,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


@dataclass
class KVAdapterConfig(ModifierConfig):
    model: str = "gpt-neo"
    soft_prompt_length: int = 10
    soft_prompt_learn_kv: bool = True
    n_tasks: int = None
    # This argument is deprecated, to ensure compatibility with `add_expert_to_transformer`
    patch_last_k_layers: int = -1


@register_modifier("kv_adapter", config_cls=KVAdapterConfig)
class KVAdapter(Adapter, ModifyMixin):
    """
    Adapter augmenting the self-attention with additional learnable KV pairs.

    Adapter modifies the self-attention call with the following execution :
    1. adapter_k, adapter_v = adapter.get_kv_weights(k_proj, v_proj)
    2. adapter_weights = adapter.route(query_states, adapter_k, self)
        2.1 `adapter.route` calls get_gate(adapter_weights)
    3. adapter_output = adapter.aggregate(adapter_weights, adapter_v)

    """

    def __init__(
        self,
        config: KVAdapterConfig,
        attn_layer: nn.Module,
    ):
        super().__init__()

        # assign self variables
        self.config = config
        self.attn_layer = attn_layer
        self.learn_kv = config.soft_prompt_learn_kv
        self.soft_prompt_length = config.soft_prompt_length
        self.soft_prompt_learn_kv = config.soft_prompt_learn_kv
        funcType = types.MethodType

        # do not patch this layer multiple times, especially useful for container layers
        if "gpt-neo" in config.model:
            self.attn_layer.forward = funcType(
                partial(gpt_neo_self_attention, adapter=self), self.attn_layer
            )
            attn_layer.hidden_size = attn_layer.embed_dim
            self.device = self.attn_layer.q_proj.weight.device
        elif "llama" in config.model:
            self.attn_layer.forward = funcType(
                partial(llama_self_attention, adapter=self), self.attn_layer
            )
            assert (
                attn_layer.num_heads == attn_layer.num_key_value_heads
            ), "which to pick for gate?"
            self.device = self.attn_layer.q_proj.weight.device
        elif "phi" in config.model:
            if "mha" in str(type(attn_layer)).lower():
                self.attn_layer.inner_cross_attn = PhiCrossAttentionModule(
                    self.attn_layer.inner_attn.causal,
                    self.attn_layer.inner_attn.softmax_scale,
                    self.attn_layer.inner_attn.drop.p,
                )
                self.attn_layer.inner_attn = PhiSelfAttentionModule(
                    self.attn_layer.inner_attn.causal,
                    self.attn_layer.inner_attn.softmax_scale,
                    self.attn_layer.inner_attn.drop.p,
                )
                self.attn_layer.inner_attn.forward = partial(
                    self.attn_layer.inner_attn.forward, adapter=self
                )
                self.attn_layer.inner_cross_attn.forward = partial(
                    self.attn_layer.inner_cross_attn.forward, adapter=self
                )
            else:
                raise ValueError(
                    f"This type of layer is not supported by KVAdapter on phi-2: {type(attn_layer)}"
                )

            assert self.soft_prompt_learn_kv, "phi only supports soft prompt kv"
            attn_layer.k_proj = attn_layer.v_proj = None
            attn_layer.num_heads = attn_layer.n_head
            attn_layer.hidden_size = attn_layer.out_proj.weight.shape[0]
            self.device = attn_layer.out_proj.weight.device
        else:
            raise ValueError(f"{config.model} not supported for now.")

        self.create_for_layer(attn_layer)

    def create_for_layer(self, attn_layer):
        if self.soft_prompt_learn_kv:
            out_dim = attn_layer.hidden_size * 2
        else:
            out_dim = attn_layer.hidden_size

        self.adapter_query = nn.Embedding(
            self.soft_prompt_length, out_dim, device=self.device
        )
        # create the gate, and embeddings here
        self.adapter_gate = torch.nn.Parameter(
            torch.zeros(1, attn_layer.num_heads, 1, 1, device=self.device),
        )

    def load_adapter_weights(self, state_dict):
        # load the weights from state_dict
        self.adapter_query.weight.data.copy_(state_dict["adapter_query.weight"])
        self.adapter_gate.data.copy_(state_dict["adapter_gate"])

    def forward(self, *args, **kwargs):
        # This Should Wrap at the SelfAttentionLevel, so behaves as such
        return self.attn_layer.forward(*args, **kwargs)

    def get_kv_weights(self, k_proj, v_proj):
        """(1) Computes the key and value pairs to be used for augmented attention"""

        if self.learn_kv:
            adapter_k, adapter_v = self.adapter_query.weight.chunk(2, dim=-1)
        else:
            adapter_k = type_safe_linear(self.adapter_query.weight, k_proj)
            adapter_v = type_safe_linear(self.adapter_query.weight, v_proj)

        out_shp = (
            1,
            self.soft_prompt_length,
            self.attn_layer.num_heads,
            self.attn_layer.head_dim,
        )
        adapter_k = adapter_k.view(*out_shp).transpose(1, 2)
        adapter_v = adapter_v.view(*out_shp).transpose(1, 2)
        return adapter_k, adapter_v

    def get_gate(self, adapter_weights):
        """Fetch the 0-init gate"""
        return self.adapter_gate

    def route(self, query, keys, attn_layer=None):
        """(2) Compute The Standard Attention Scores in augmented attention"""
        if attn_layer is None:
            attn_layer = self.attn_layer

        adapter_logits = torch.matmul(
            query, keys.transpose(2, 3).type_as(query)
        ) / math.sqrt(attn_layer.head_dim)

        adapter_weights = F.softmax(adapter_logits, dim=-1, dtype=torch.float32)
        gate_out = self.get_gate(adapter_weights)
        out = gate_out * adapter_weights.type_as(query)
        return out

    def aggregate(self, adapter_weights, adapter_v):
        """(3) Aggregate the weighted values according to the adapter weights"""
        return torch.matmul(adapter_weights, adapter_v)


""" """ """ """
""" Model Specific Implementations of Self Attention """
""" """ """ """


def type_safe_linear(input, linear_layer):
    dtype_input = input.dtype
    input = input.type(linear_layer.weight.dtype)
    output = linear_layer(input)
    return output.type(dtype_input)


def llama_self_attention(
    self,
    hidden_states: torch.Tensor,
    adapter: KVAdapter = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz, q_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        # `past_key_value` should always also cache the adapter k,v
        past_k, past_v, adapter_k, adapter_v = past_key_value
        kv_seq_len += past_k.size(-2)  #  + adapter_k.size(-2)
    else:
        adapter_k = adapter_v = None

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        key_states = torch.cat([past_k, key_states], dim=2)
        value_states = torch.cat([past_v, value_states], dim=2)

    # Typically saved here, before repeat_kv. if `num_key_value_groups` > 1, need to adjust
    # past_key_value = (key_states, value_states) if use_cache else None

    assert self.num_key_value_groups == 1, "how to handle this for adapter?"
    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        self.head_dim
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    """ Adapter Start """
    # adapter not precomputed, so we compute it
    if adapter_k is None:
        adapter_k, adapter_v = adapter.get_kv_weights(self.k_proj, self.v_proj)
    if use_cache:
        past_key_value = (key_states, value_states, adapter_k, adapter_v)

    adapter_weights = adapter.route(query_states, adapter_k, self)
    adapter_output = adapter.aggregate(adapter_weights, adapter_v).type_as(attn_output)
    """ Adapter End  """

    # merge and reshape
    attn_output = attn_output + adapter_output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    # NOTE: not applying positional embedding on these ones.
    assert self.config.pretraining_tp == 1, "see `modeling_llama.py` to add support"
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def gpt_neo_self_attention(
    self,
    hidden_states,
    adapter: KVAdapter = None,
    attention_mask=None,
    layer_past=None,
    head_mask=None,
    use_cache=False,
    output_attentions=False,
):
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    adapter_k = adapter_v = present = None

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
        adapter_k, adapter_v = layer_past[2], layer_past[3]

    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

    """ Adapter Start """
    # adapter not precomputed, so we compute it
    if adapter_k is None:
        adapter_k, adapter_v = adapter.get_kv_weights(self.k_proj, self.v_proj)
    if use_cache:
        present = (key, value, adapter_k, adapter_v)

    # remains to compute the attention score and add the result
    adapter_weights = adapter.route(query, adapter_k, self)
    adapter_output = adapter.aggregate(adapter_weights, adapter_v).type_as(attn_output)
    attn_output = attn_output + adapter_output
    """ Adapter End """

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


class PhiCrossAttentionModule(nn.Module):
    def __init__(self, causal, softmax_scale, attention_dropout):
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        q: torch.FloatTensor,
        kv: torch.FloatTensor,
        adapter: KVAdapter = None,
        causal: bool = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = kv.shape[1]

        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k),
                -10000.0,
                dtype=scores.dtype,
                device=scores.device,
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            rows = rearrange(
                torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1"
            )
            cols = torch.arange(seqlen_k, device=k.device, dtype=torch.long)
            causal_mask = cols > rows + seqlen_k - seqlen_q

            scores = scores.masked_fill(causal_mask, -10000.0)

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.drop(attention)
        attn_output = torch.einsum("bhts,bshd->bthd", attention, v)

        """ Adapter Start """
        # adapter not precomputed, so we compute it
        adapter_k, adapter_v = adapter.get_kv_weights(None, None)

        # remains to compute the attention score and add the result
        adapter_weights = adapter.route(q.transpose(1, 2), adapter_k)
        adapter_output = adapter.aggregate(adapter_weights, adapter_v).type_as(
            attn_output
        )
        attn_output = attn_output + adapter_output.transpose(1, 2)
        """ Adapter End """

        return attn_output


class PhiSelfAttentionModule(nn.Module):
    def __init__(self, causal, softmax_scale, attention_dropout):
        super().__init__()

        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)

    def forward(
        self,
        qkv: torch.FloatTensor,
        causal: bool = None,
        adapter: KVAdapter = None,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)

        causal = self.causal if causal is None else causal
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)

            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")

        if causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)

        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.drop(attention)

        attn_output = torch.einsum("bhts,bshd->bthd", attention, v)

        """ Adapter Start """
        # adapter not precomputed, so we compute it
        adapter_k, adapter_v = adapter.get_kv_weights(None, None)

        # remains to compute the attention score and add the result
        adapter_weights = adapter.route(q.transpose(1, 2), adapter_k)
        adapter_output = adapter.aggregate(adapter_weights, adapter_v).type_as(
            attn_output
        )
        attn_output = attn_output + adapter_output.transpose(1, 2)
        """ Adapter End """

        return attn_output
