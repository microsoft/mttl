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
from mttl.models.modifiers.base import Adapter, ModifierConfig, ModifyMixin
from mttl.models.modifiers.poly import PolytroponSelector, PolytroponConfig


from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaForCausalLM,
)


def llama_adapter_attention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    adapter: Optional[torch.Tensor] = None,
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

    def type_safe_linear(input, linear_layer):
        dtype_input = input.dtype
        input = input.type(linear_layer.weight.dtype)
        output = linear_layer(input)
        return output.type(dtype_input)

    """ Adapter Start """
    # adapter not precomputed, so we compute it
    if adapter_k is None:
        adapter = self.adapter()
        if self.adapter.learn_kv:
            adapter_k, adapter_v = adapter.chunk(2, dim=-1)
        else:
            adapter_k = type_safe_linear(adapter, self.k_proj)
            adapter_v = type_safe_linear(adapter, self.v_proj)

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

    adapter_weights = self.adapter.adapter_gate * F.softmax(
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


def modify_with_llama_adapters(cls, transformer, config, **kwargs):
    assert isinstance(
        transformer, PreTrainedModel
    ), "Transformer must be a PreTrainedModel."

    assert isinstance(
        transformer, LlamaForCausalLM
    ), "Only Llama Model supported for now."
    task_id_ptr = transformer.task_id_container

    for param in transformer.parameters():
        param.requires_grad = False

    def wrap_llama_attn(attn_layer):
        # create the prefix embeddings
        attn_layer.soft_prompt_length = config.soft_prompt_length
        attn_layer.forward = partial(llama_adapter_attention, attn_layer)
        attn_layer.adapter = cls(config, attn_layer, task_id_ptr, **kwargs)
        assert (
            attn_layer.num_heads == attn_layer.num_key_value_heads
        ), "which to pick for gate?"

    if config.patch_last_k_layers == -1:
        layers_to_patch = transformer.model.layers
    else:
        layers_to_patch = [transformer.model.layers[-config.patch_last_k_layers]]

    for layer in layers_to_patch:
        wrap_llama_attn(layer.self_attn)

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


@dataclass
class PolyLLamaAdapterConfig(LLamaAdapterConfig, PolytroponConfig):
    n_skills: int = 1
    n_splits: int = 1


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