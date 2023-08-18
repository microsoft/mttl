
import math
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, List
import re
import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target_name, target

def modify_with_llama_adapter(transformer, config):
    parents=[]                   
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.adapter_modules, m_name.split(".")[-1]):
            par, _, _ = _get_submodules(transformer, m_name)
            parents.append(par)
    parents = parents[-config.adapter_layers :]
    _create_adapted_attentions(config, parents)
    return transformer


def _create_adapted_attentions(config,  parents: List[nn.Module]) -> None:
    """Wrap LlamaAttention modules with newly created AdaptedAttention modules."""
    for par in parents:  
        attn = AdaptedAttention( 
            model_type=config.model, #"llama",
            adapter_len=config.adapter_len,  
            model=getattr(par, config.adapter_modules),
        )
        setattr(par, config.adapter_modules, attn)

def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.requires_grad = False

def llama_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    This function was duplicated verbatim from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L126
    This was done to eliminate the Llama transformers implementation as a dependency of this file. Note that some other
    functions were also adapted from the transformers implementation but were modified.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def llama_apply_rotary_pos_emb(q, cos, sin, position_ids):
    """
    Apply rotary position embedding to query states in the Llama model.
    This function was adapted from:
    https://github.com/huggingface/transformers/blob/1de8ce9ee1191ba761a593ac15d9ccbf5851bfc5/src/transformers/models/llama/modeling_llama.py#L133
    It was modified to remove unnecessary processing of key states.
    """
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (llama_rotate_half(q) * sin)
    return q_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def gptneox_apply_rotary_pos_emb(q, cos, sin, position_ids):
    """    
    Apply rotary position embedding to query states in the GPTNeoX model.
    This function was adapted from:
    https://github.com/huggingface/transformers/blob/db136341836d6d9a4d599e2935253e0ae04cc1f1/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L287
    It was modified to remove unnecessary processing of key states.
    """

    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed


def llama_compute_query_states(model: nn.Module, **kwargs) -> torch.Tensor:
    """
    Compute query states for Llama models specifically.
    They need to be recomputed as the forward() method of the original LlamaModel in the transformers library does not
    return them. See the related discussion in the PR: https://github.com/huggingface/peft/pull/268
    """
    hidden_states = kwargs.get("hidden_states")
    position_ids = kwargs.get("position_ids")
    past_key_value = kwargs.get("past_key_value")
    bsz, q_len, _ = hidden_states.size()
    query_states = model.q_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)
    value_states = model.v_proj(hidden_states).view(bsz, q_len, model.num_heads, model.head_dim).transpose(1, 2)

    seq_len = q_len
    if past_key_value is not None:
        seq_len += past_key_value[0].shape[-2]
    cos, sin = model.rotary_emb(value_states, seq_len=seq_len)

    return llama_apply_rotary_pos_emb(query_states, cos, sin, position_ids)

def gptneox_compute_query_states(model: nn.Module, **kwargs) -> torch.Tensor:
    """
    Compute query states.
    They need to be recomputed as the forward() method of the original LlamaModel in the transformers library does not
    return them. See the related discussion in the PR: https://github.com/huggingface/peft/pull/268
    """
    hidden_states = kwargs.get("hidden_states")
    position_ids = kwargs.get("position_ids")
    past_key_value = kwargs.get("past_key_value")
    bsz, q_len, _ = hidden_states.size()  
    num_heads = model.num_heads if hasattr(model, "num_heads") else model.num_attention_heads
    head_dim = model.head_dim if hasattr(model, "head_dim") else model.head_size
    query_states, _, value_states = model.query_key_value(hidden_states).split(model.hidden_size, dim=-1)
    query_states=query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    seq_len = q_len
    if past_key_value is not None:
        seq_len += past_key_value[0].shape[-2]
    cos, sin = model.rotary_emb(value_states, seq_len=seq_len)

    return gptneox_apply_rotary_pos_emb(query_states, cos, sin, position_ids)


# Contains the config that is specific to a transformers model type.
ModelTypeConfig = namedtuple(
    "ModelTypeConfig", ["compute_query_states", "target_modules", "k_proj_layer", "v_proj_layer", "o_proj_layer", "dtype"]
)
# Mapping of transformers model types to their specific configuration.
TRANSFORMERS_MODEL_CONFIG = {  
    "yahma/llama-7b-hf": ModelTypeConfig(      
        compute_query_states=llama_compute_query_states,
        target_modules="self_attn",
        k_proj_layer="k_proj",
        v_proj_layer="v_proj",
        o_proj_layer="o_proj",
        dtype=torch.float32,
    ),
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1": ModelTypeConfig(
        compute_query_states=gptneox_compute_query_states,
        target_modules="self_attn",  
        k_proj_layer="query_key_value",
        v_proj_layer="query_key_value",
        o_proj_layer="dense",
        dtype=torch.float32,
    ),
}
class AdaptedAttention(nn.Module):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: str, adapter_len: int, model):#, model_config=None):
        """
        Initialize object.
        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model      
        self.adapter_len = adapter_len
        # self.model_config = model_config
        # Assume all parameters of the attention model we are wrapping are on the same device.
        device = next(model.parameters()).device
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)
        self.adaption_prompt = nn.Parameter(  
            torch.empty(1, adapter_len, self.model.hidden_size, device=device, dtype=TRANSFORMERS_MODEL_CONFIG[self.model_type].dtype).normal_()
        )
        # Initialize the gate to 0 as this is "zero-init".      
        self.adaption_gate = nn.Parameter(torch.zeros(1, device=device, dtype=TRANSFORMERS_MODEL_CONFIG[self.model_type].dtype))
        # if not self.model_type in ['yahma/llama-7b-hf']:
            # model_config = {"hidden_size":self.model.head_size, "num_heads":self.model.num_attention_heads, "head_dim":self.model.head_size, "max_position_embeddings": self.model.}
            # new_attention_layer = LlamaAttention(self.model_config)
            # print("Adapting layer of", self.model_type)

    def forward(self, *args, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.
        "Official" paper implementation:
        https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L141
        Args:
            kwargs: See the original LlamaAttention module.
        """
        if kwargs.get("output_attention", False):
            raise NotImplementedError("output_attention is not currently supported.")
        if "hidden_states" not in kwargs:
            kwargs["hidden_states"] = args[0]
        output = self.model(**kwargs)
        if len(output) == 3:
            output, _, past_key_value = output
        else:
            output, past_key_value = output
        bsz = output.shape[0]
        q_len = output.shape[1]
        embed_dim = output.shape[2]
        # TODO: adapt this for GPTNeoX
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer

        if k_proj_layer == v_proj_layer:  
            _, key, value = getattr(self.model, k_proj_layer)(self.adaption_prompt).split(embed_dim, dim=2)
        else:
            key = getattr(self.model, k_proj_layer)(self.adaption_prompt)
            value = getattr(self.model, v_proj_layer)(self.adaption_prompt)
        # (bsz, num_heads, adapter_len, head_dim)
        num_heads = self.model.num_heads if hasattr(self.model, "num_heads") else self.model.num_attention_heads
        head_dim = self.model.head_dim if hasattr(self.model, "head_dim") else self.model.head_size
        adapter_k = (
            key.view(1, self.adapter_len, num_heads, head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )
        # (bsz, num_heads, adapter_len, head_dim)
        adapter_v = (
            value.view(1, self.adapter_len, num_heads, head_dim)
            .repeat(bsz, 1, 1, 1)
            .transpose(1, 2)
        )

        # Recompute query states.
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        # (bsz, num_heads, q_len, head_dim)
        query_states = compute_query_states(model=self.model, **kwargs)

        # (bsz, num_heads, q_len, adapter_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3)) / math.sqrt(num_heads)
        # Upcast attention to fp32
        # (bsz, num_heads, q_len, adapter_len)
        scores = self.adaption_gate * F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # (bsz, q_len, num_heads * head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)
        # (bsz, q_len, hidden_size)
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        # Add adaption prompt output to original output.
        output = output + adapter_output
        if self.model_type=="togethercomputer/RedPajama-INCITE-Base-7B-v0.1":
            return output, past_key_value
        return output, None, past_key_value