from dataclasses import dataclass
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mttl.models.modifiers import register_modifier
from transformers.modeling_utils import PreTrainedModel
from functools import partial
from typing import Optional, Tuple
from mttl.utils import logger
from mttl.models.modifiers.base import Adapter, ModifierConfig, ModifyMixin
from mttl.models.modifiers.poly import PolytroponSelector, PolytroponConfig
from mttl.models.modifiers.base import MergeableAdapter, ModifyMixin, ModifierConfig

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaForCausalLM,
)
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM


@dataclass
class KVAdapterConfig(ModifierConfig):
    soft_prompt_length: int = 10
    soft_prompt_learn_kv: bool = True
    n_tasks: int = None
    # This argument is deprecated, as control has been switched to
    patch_last_k_layers: int = -1


@dataclass
class PolyKVAdapterConfig(KVAdapterConfig, PolytroponConfig):
    n_skills: int = 1
    n_splits: int = 1


@register_modifier("kv_adapter", config_cls=KVAdapterConfig)
class KVAdapter(MergeableAdapter, ModifyMixin):
    def __init__(
        self,
        config: KVAdapterConfig,
        attn_layer: nn.Module,
        task_id_ptr: dict,
    ):
        super().__init__()

        # assign self variables
        self.config = config
        self.attn_layer = attn_layer
        self.task_id_ptr = task_id_ptr
        self.learn_kv = config.soft_prompt_learn_kv
        self.soft_prompt_length = config.soft_prompt_length
        self.soft_prompt_learn_kv = config.soft_prompt_learn_kv
        attn_layer.soft_prompt_length = config.soft_prompt_length

        if "gpt-neo" in config.model:
            attn_layer.hidden_size = attn_layer.embed_dim
            self.attn_fwd = gpt_neo_self_attention
        elif "llama" in config.model:
            self.attn_fwd = llama_self_attention
            assert (
                attn_layer.num_heads == attn_layer.num_key_value_heads
            ), "which to pick for gate?"
        else:
            raise ValueError(f"{config.model} not supported for now.")

        self.create_for_layer(attn_layer)

    def create_for_layer(self, attn_layer):
        # create the gate, and embeddings here
        self.adapter_gate = torch.nn.Parameter(
            torch.zeros(1, attn_layer.num_heads, 1, 1)
        )

        if self.soft_prompt_learn_kv:
            out_dim = attn_layer.hidden_size * 2
        else:
            out_dim = attn_layer.hidden_size

        self.adapter_query = nn.Embedding(self.soft_prompt_length, out_dim)

    def load_adapter_weights(self, state_dict):
        # load the weights from state_dict
        self.adapter_query.weight.data.copy_(state_dict["adapter_query.weight"])
        self.adapter_gate.data.copy_(state_dict["adapter_gate"])

    def merge_with_layer(self):
        # TODO: Should I just remove the `MergeableAdapter` inheritance ?
        logger.warning("cannot merge `KVAdapter` with base layer. No-op")

    def forward(self, *args, **kwargs):
        # This Should Wrap at the SelfAttentionLevel, so behaves as such
        return self.attn_fwd(self.attn_layer, self, None, *args, **kwargs)

    def get_adapter(self, expand=True):
        bsz = self.task_id_ptr["routing_infos"].task_ids.size(0)
        out = self.adapter_query.weight.unsqueeze(0)
        return out.expand(bsz, -1, -1) if expand else out

    @classmethod
    def parallel_linear_forward(cls, input, kv_adapters, **kwargs):
        if len(set([kv_adapter.attn_layer for kv_adapter in kv_adapters])) > 1:
            raise ValueError("Cannot parallelize adapters applied to different layers.")

        # Build soft prompts
        adapters = [kv_adapter.get_adapter(expand=False) for kv_adapter in kv_adapters]
        adapters = torch.cat(adapters)
        kv_adapter, attn_layer = kv_adapters[0], kv_adapters[0].attn_layer
        return kv_adapter.attn_fwd(attn_layer, kv_adapter, adapters, input, **kwargs)

    @classmethod
    def modify_transformer(cls, transformer, config, **kwargs):
        """Wrap attention layer with additonal tokens"""
        return modify_with_kv_adapters(cls, transformer, config, **kwargs)


def type_safe_linear(input, linear_layer):
    dtype_input = input.dtype
    input = input.type(linear_layer.weight.dtype)
    output = linear_layer(input)
    return output.type(dtype_input)


def llama_self_attention(
    self,
    adapter: KVAdapter,
    adapter_weights: torch.Tensor,
    hidden_states: torch.Tensor,
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
        if adapter_weights is None:
            adapter_weights = adapter.get_adapter()
        if adapter.learn_kv:
            adapter_k, adapter_v = adapter_weights.chunk(2, dim=-1)
        else:
            adapter_k = type_safe_linear(adapter_weights, self.k_proj)
            adapter_v = type_safe_linear(adapter_weights, self.v_proj)

        adapter_k = adapter_k.view(
            bsz, self.soft_prompt_length, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # bs, n_heads, len, d_head
        adapter_v = adapter_v.view(
            bsz, self.soft_prompt_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

    if use_cache:
        past_key_value = (key_states, value_states, adapter_k, adapter_v)
    else:
        past_key_value = None

    adapter_weights = torch.matmul(
        query_states, adapter_k.transpose(2, 3).type_as(query_states)
    ) / math.sqrt(self.head_dim)

    adapter_weights = adapter.adapter_gate * F.softmax(
        adapter_weights, dim=-1, dtype=torch.float32
    ).type_as(query_states)

    adapter_output = torch.matmul(adapter_weights, adapter_v).type_as(attn_output)
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
    adapter,
    adapter_weights,
    hidden_states,
    attention_mask=None,
    layer_past=None,
    head_mask=None,
    use_cache=False,
    output_attentions=False,
):
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    adapter_k = adapter_v = None

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)

    bsz, n_heads, seq_len, d_head = query.size()

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
        if adapter_weights is None:
            adapter_weights = adapter.get_adapter()
        if adapter.learn_kv:
            adapter_k, adapter_v = adapter_weights.chunk(2, dim=-1)
        else:
            adapter_k = type_safe_linear(adapter_weights, self.k_proj)
            adapter_v = type_safe_linear(adapter_weights, self.v_proj)

        adapter_k = adapter_k.view(
            bsz, self.soft_prompt_length, self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # bs, n_heads, len, d_head
        adapter_v = adapter_v.view(
            bsz, self.soft_prompt_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    # remains to compute the attention score and add the result
    adapter_weights = torch.matmul(
        query, adapter_k.transpose(2, 3).type_as(query)
    ) / math.sqrt(self.head_dim)

    adapter_weights = adapter.adapter_gate * F.softmax(
        adapter_weights, dim=-1, dtype=torch.float32
    ).type_as(query)

    adapter_output = torch.matmul(adapter_weights, adapter_v).type_as(attn_output)
    attn_output = attn_output + adapter_output
    """ Adapter End """

    if use_cache is True:
        present = (key, value, adapter_k, adapter_v)
    else:
        present = None

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # a, present, (attentions)


def modify_with_kv_adapters(adapter_class, transformer, config, **kwargs):
    assert isinstance(
        transformer, PreTrainedModel
    ), "Transformer must be a PreTrainedModel."

    task_id_ptr = transformer.task_id_container

    for param in transformer.parameters():
        param.requires_grad = False

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.modify_layers, c_name):
                    logger.info(f"Patching {m_name}.{c_name}...")

                    setattr(
                        module,
                        c_name,
                        adapter_class(config, layer, task_id_ptr),
                    )

    return transformer


'''
def modify_with_llama_adapters(cls, transformer, config, **kwargs):
    assert isinstance(
        transformer, PreTrainedModel
    ), "Transformer must be a PreTrainedModel."

    def wrap_llama_attn(attn_layer):
        # create the prefix embeddings
        attn_layer.soft_prompt_length = config.soft_prompt_length
        attn_layer.forward = partial(llama_self_attention, attn_layer)
        attn_layer.adapter = cls(config, attn_layer, task_id_ptr, **kwargs)
        assert (
            attn_layer.num_heads == attn_layer.num_key_value_heads
        ), "which to pick for gate?"

    def wrap_gpt_neo_attn(attn_layer):
        # create the prefix embeddings
        attn_layer.soft_prompt_length = config.soft_prompt_length
        attn_layer.forward = partial(gpt_neo_self_attention, attn_layer)
        attn_layer.adapter = cls(config, attn_layer, task_id_ptr, **kwargs)

    if isinstance(transformer, LlamaForCausalLM):
        wrap_fn = wrap_llama_attn
        attn_layers = [layer.self_attn for layer in transformer.model.layers]
    elif isinstance(transformer, GPTNeoForCausalLM):
        wrap_fn = wrap_gpt_neo_attn
        attn_layers = [layer.attn.attention for layer in transformer.transformer.h]
        # `hidden_size` needs to be set
        for attn_layer in attn_layers:
            attn_layer.hidden_size = attn_layer.embed_dim

    task_id_ptr = transformer.task_id_container

    for param in transformer.parameters():
        param.requires_grad = False

    if config.patch_last_k_layers != -1:
        attn_layers = attn_layers[-config.patch_last_k_layers]

    for layer in attn_layers:
        wrap_fn(layer)

    return transformer


@dataclass
class LLamaAdapterConfig(ModifierConfig):
    soft_prompt_length: int = 10
    soft_prompt_learn_kv: bool = False  # should we set this to True?
    n_tasks: int = None
    patch_last_k_layers: int = -1


@register_modifier("llama_adapter", config_cls=LLamaAdapterConfig)
class LlamaAdapter(nn.Module):
    def __init__(self, config, attn_layer, task_id_ptr, **kwargs):
        super().__init__()

        self.task_id_ptr = task_id_ptr
        self.adapter_gate = torch.nn.Parameter(
            torch.zeros(1, attn_layer.num_heads, 1, 1)
        )
        self.learn_kv = config.soft_prompt_learn_kv

        if config.soft_prompt_learn_kv:
            out_dim = attn_layer.hidden_size * 2
        else:
            out_dim = attn_layer.hidden_size

        self.adapter_query = nn.Embedding(config.soft_prompt_length, out_dim)

    # get adapter
    def forward(self):
        bsz = self.task_id_ptr["routing_infos"].task_ids.size(0)
        return self.adapter_query.weight.unsqueeze(0).expand(bsz, -1, -1)

    @classmethod
    def modify_transformer(cls, transformer, config, **kwargs):
        """Wrap attention layer with additonal tokens"""
        return modify_with_llama_adapters(cls, transformer, config, **kwargs)


@dataclass
class MLPLLamaAdapterConfig(LLamaAdapterConfig):
    soft_prompt_mlp_dim: int = 128


@register_modifier("mlp_llama_adapter", config_cls=MLPLLamaAdapterConfig)
class MLPLlamaAdapter(Adapter, ModifyMixin):
    def __init__(self, mlp, config, attn_layer, task_id_ptr, **kwargs):
        super().__init__()

        self.task_id_ptr = task_id_ptr
        self.learn_kv = config.soft_prompt_learn_kv
        self.adapter_gate = torch.nn.Parameter(
            torch.zeros(1, attn_layer.num_heads, 1, 1)
        )
        self.adapter_query = nn.Embedding(
            config.soft_prompt_length, config.soft_prompt_mlp_dim
        )

        self.shared_adapter_mlp = mlp

    # get adapter
    def forward(self):
        bsz = self.task_id_ptr["routing_infos"].task_ids.size(0)
        mlp_input = self.adapter_query.weight
        out = self.shared_adapter_mlp(mlp_input)
        return out.unsqueeze(0).expand(bsz, -1, -1)

    @classmethod
    def modify_transformer(cls, transformer, config):
        if config.soft_prompt_learn_kv:
            out_dim = transformer.config.hidden_size * 2
        else:
            out_dim = transformer.config.hidden_size

        shared_adapter_mlp = nn.Sequential(
            nn.Linear(config.soft_prompt_mlp_dim, config.soft_prompt_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.soft_prompt_hidden_dim, out_dim),
        )

        return modify_with_llama_adapters(
            cls, transformer, config, mlp=shared_adapter_mlp
        )


@register_modifier("poly_llama_adapter", config_cls=PolyLLamaAdapterConfig)
class PolyLlamaAdapter(Adapter, ModifyMixin):
    def __init__(self, config, attn_layer, task_id_ptr, **kwargs):
        super().__init__()

        self.selector = PolytroponSelector(config)
        self.learn_kv = config.soft_prompt_learn_kv
        self.n_skills, self.n_splits = config.n_skills, config.n_splits
        self.prompt_len = config.soft_prompt_length
        self.task_id_ptr = task_id_ptr

        if config.soft_prompt_learn_kv:
            out_dim = attn_layer.hidden_size * 2
        else:
            out_dim = attn_layer.hidden_size

        self.adapter_query = nn.Embedding(
            config.n_skills * config.soft_prompt_length, out_dim
        )
        self.adapter_gate = torch.nn.Parameter(
            torch.zeros(1, attn_layer.num_heads, 1, 1)
        )

    # get adapter
    def forward(self):
        weight = self.adapter_query.weight.reshape(
            self.n_skills, self.prompt_len, self.n_splits, -1
        )
        mixing_weights = self.selector(self.task_id_ptr["routing_infos"])
        out = torch.einsum("klsd,bsk->blsd", weight, mixing_weights)

        return out.flatten(-2)

    @classmethod
    def modify_transformer(cls, transformer, config):
        return modify_with_llama_adapters(cls, transformer, config)
'''
