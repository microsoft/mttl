import inspect
import os
from typing import Optional, Tuple

try:
    import flash_attn
except ImportError:
    flash_attn = None

import torch
import torch.nn.functional as F

from mttl.logging import warn_once
from mttl.models.expert_context import InfoContainer

""" Pytorch SDPA Patching """


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    from mttl.models.expert_context import InfoContainer

    context = InfoContainer.get()
    if context is not None and context._routing_infos.packed_seq_lens is not None:
        attn_mask = context._routing_infos.packed_attn_mask
        is_causal = False

    return torch.nn.functional._default_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


""" Flash Attention Patching """


def flash_attn_varlen_func_wrapper(
    query_states,
    key_states,
    value_states,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    causal,
    **flash_kwargs,
):
    if query_states.shape != key_states.shape:
        raise ValueError("q and k must have the same shape")

    context = InfoContainer.get()
    if context is not None and context.routing_infos.packed_seq_lens is not None:
        warn_once(
            "\n\n\n\nUsing the Flash Attention 2 Sequence Packing Wrapper\n\n\n\n"
        )
        cu_seqlens_q = context.routing_infos.packed_seq_lens
        cu_seqlens_k = context.routing_infos.packed_seq_lens
        max_seqlen_q = context.routing_infos.seq_lens.max().item()
        max_seqlen_k = context.routing_infos.seq_lens.max().item()

    return flash_attn._default_flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        **flash_kwargs,
    )


def flash_attn_func_wrapper(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):

    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")

    # assert there are no padding tokens if we get here
    context = InfoContainer.get()
    assert (context.routing_infos.attention_mask == 1).all()  # no padding tokens

    if context.routing_infos.packed_seq_lens is not None:
        cu_seqlens_q = cu_seqlens_k = context.routing_infos.packed_seq_lens
        max_seqlen_q = max_seqlen_k = context.routing_infos.seq_lens.max().item()
        q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)

        return flash_attn_varlen_func_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
    else:
        return flash_attn._default_flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )
