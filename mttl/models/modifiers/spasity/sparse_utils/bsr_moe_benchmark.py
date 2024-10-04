import logging
import re
import time
from typing import List

import numpy as np
import pandas as pd
import stk.ops
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton as tn
from pytorch_lightning import seed_everything
from spops import csr_add, spmm
from stk.matrix import Matrix
from triton.ops.blocksparse import matmul

from mttl.logging import logger
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import Modifier
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRA, SkilledLoRAConfig
from mttl.models.modifiers.spasity.sparse_mask import (
    MaskedLinear,
    ScatteredSparseLinearModule,
    SparseLinearModule,
    SparseMaskAdapter,
    SparseMaskConfig,
)
from mttl.models.modifiers.spasity.sparse_utils.utils import (
    padded_gather,
    padded_scatter,
)
from mttl.models.modifiers.spasity.spb_moe import _matrix_ops, linear_ops
from mttl.models.utils import model_loader_helper, transfer_batch_to_device

device = "cuda"
logger.setLevel(logging.ERROR)
block_size = 16  # 128  # 16
n_blocks = 1024  # 16  # 1024
in_d = 2048
out_d = 8192
dtype = torch.bfloat16
max_seq_len = 1024
bs = 2
layer = nn.Linear(in_d, out_d).to(device)
layer.weight.requires_grad_(False)
layer.bias.requires_grad_(False)
K = 100
top_k = 2


def calculate_lora_parameters(input_dim, output_dim, rank):
    return input_dim * rank + output_dim * rank


def find_hyperpaams():
    modules = {"linear": layer}
    modified_modules = {}
    keep_ratios = []
    lora_ranks = []

    for name, module in modules.items():
        keep_ratio = (
            n_blocks * (block_size**2) / (module.in_features * module.out_features)
        )
        tot_sparse_params = module.in_features * module.out_features * keep_ratio
        lora_rank = 1
        for rank in range(1, module.in_features):
            lora_params = calculate_lora_parameters(
                module.in_features, module.out_features, rank
            )
            if lora_params <= tot_sparse_params:
                lora_rank = rank
            else:
                break
        modified_modules[name] = {
            "module": module,
            "keep_ratio": keep_ratio,
            "lora_rank": lora_rank,
        }
        keep_ratios.append(keep_ratio)
        lora_ranks.append(lora_rank)
    return np.mean(keep_ratios), int(np.mean(lora_ranks))


keep_ratio, lora_rank = find_hyperpaams()
print(
    f"Keep ratio: {keep_ratio}, LoRA rank: {lora_rank}, Lora params: {calculate_lora_parameters(in_d, out_d, lora_rank)}, Sparse params: {in_d * out_d * keep_ratio}"
)
x = torch.randn(bs, max_seq_len, in_d, dtype=dtype, device=device).contiguous()


def create_adapter_set(adapter_config, layer, K) -> List[Modifier]:
    if isinstance(adapter_config, SparseMaskConfig):
        layer = nn.Linear(out_d, in_d)  # TODO: implement transpose in SparseWeights
        module = [SparseMaskAdapter(adapter_config, layer) for _ in range(K)]
    elif isinstance(adapter_config, LoRAConfig):
        module = [LoRA(adapter_config, layer) for _ in range(K)]
    return module


def sparsemodules_to_stkmatrix_list(sparse_modules):
    sparse_weights = []
    for sparse_module in sparse_modules:
        mtx = stk.ops.to_sparse(
            sparse_module.sparse_layer.to_dense().type(dtype), blocking=block_size
        )
        # mtx.validate()
        sparse_weights.append(mtx)
    return sparse_weights


@torch.autocast(device_type="cuda", dtype=dtype)
def lora_merge(lora_a, lora_b, x, W_base, W_merge):

    # merge into 1 loa
    A = torch.einsum("ble,edr->bldr", (W_merge, lora_a))
    B = torch.einsum("ble,erd->blrd", (W_merge, lora_b))
    # lora forward
    partial_out = torch.einsum("bld,bldr->blr", (x, A))
    adapter_out = torch.einsum("blr,blrd->bld", (partial_out, B))
    dense_out = x @ W_base
    return adapter_out + dense_out


def create_block_diagonal_matrix(bs_m, bs_n, n_blocks):
    assert bs_m >= block_size
    assert bs_n >= block_size
    factor = (bs_m * bs_n) // (block_size**2)

    M = bs_m * n_blocks
    N = bs_n * n_blocks

    Mb = M // block_size
    Nb = N // block_size

    nb_m_pb = bs_m // block_size
    nb_n_pb = bs_n // block_size

    col_indices_1blk = torch.arange(nb_n_pb, device=device, dtype=torch.int32).repeat(
        nb_m_pb
    )
    row_indices_1blk = torch.arange(
        nb_m_pb, device=device, dtype=torch.int32
    ).repeat_interleave(nb_n_pb)
    offsets = torch.arange(0, Mb * nb_n_pb + nb_n_pb, nb_n_pb, device=device)

    col_idx = torch.cat([col_indices_1blk + i * nb_n_pb for i in range(n_blocks)])
    row_idx = torch.cat([row_indices_1blk + i * nb_m_pb for i in range(n_blocks)])
    data = torch.empty((Mb * Nb, block_size, block_size), device=device)

    return Matrix((M, N), data, row_idx, col_idx, offsets)


adapter_config_lora = LoRAConfig(modify_layers="", lora_rank=lora_rank)
adapter_config_bs = SparseMaskConfig(
    sps_impl="scattered",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)

# FOWARD PASS through MoE
W_mege = torch.randn(bs, max_seq_len, K, dtype=dtype, device=device)
loras = create_adapter_set(adapter_config_lora, layer, K)
sparse_modules = create_adapter_set(adapter_config_bs, layer, K)
sparse_mtxs = sparsemodules_to_stkmatrix_list(sparse_modules)
adaptersMatrix: Matrix = _matrix_ops.merge_adapters(sparse_mtxs).to(device)

W_mege = W_mege.to(dtype=loras[0].lora_a.dtype)
top_k_indices = torch.topk(torch.abs(W_mege), top_k, dim=-1).indices
(
    x,
    num_tokens_per_expert,
    sort_order,
    indices_expert_padded,
    positions_in_expert_padded,
    padding_mask,
) = padded_gather(x, top_k_indices, K)
layout = _matrix_ops.create_ada_layout(adaptersMatrix).to(device)

out_blck_size = x.shape[1]
x = x.reshape(-1, in_d).contiguous()
out_topology = create_block_diagonal_matrix(out_blck_size, out_d, K)
W_base = layer.weight.T.to(dtype=dtype)
output = linear_ops.sdd_adamerge(x, W_base, out_topology, adaptersMatrix, layout)
print(output.shape)
# create output topoly


# @tn.testing.perf_report(
#     tn.testing.Benchmark(
#         x_names=["K"],  # Argument names to use as an x-axis for the plot.
#         x_vals=[2, 3, 4, 10, 64, 128],  # Different possible values for `x_name`.
#         x_log=False,  # x axis is logarithmic.
#         line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
#         line_vals=[
#             "lora",
#         ],  # "lora_compiled", "torch_sparse_compiled"],  # Possible values for `line_arg`.
#         line_names=[
#             "lora",
#         ],  # "lora_compiled", "torch_sparse_compiled"],  # Label name for the lines.
#         styles=[
#             ("blue", "-"),
#             ("green", "-"),
#             ("orange", "-"),
#             ("red", "-"),
#             ("purple", "-"),
#             ("black", "-"),
#             ("brown", "-"),
#         ],  # Line color and style.
#         ylabel="ms",  #'GB/s',  # Label name for the y-axis.
#         xlabel="K",
#         plot_name="matmul-performance",  # Name for the plot. Used also as a file name for saving the plot.
#         args={"bs": bs, "max_seq_len": max_seq_len, "in_d": in_d, "d_out": out_d},
#     )
# )
# def benchmark(K, bs, max_seq_len, in_d, d_out, provider):
#     W_mege = torch.randn(bs, max_seq_len, K, dtype=dtype, device=device)
#     loras = create_adapter_set(adapter_config_lora, layer, K)
#     sparse_modules = create_adapter_set(adapter_config_bs, layer, K)
#     W_mege = W_mege.to(dtype=loras[0].lora_a.dtype)

#     lora_a = torch.stack([lora.lora_a for lora in loras], dim=0)
#     lora_b = torch.stack([lora.lora_b for lora in loras], dim=0)
#     sparse_weights: List[torch.Tensor] = [
#         sparse_module.sparse_layer.to_dense().to_sparse_csr().to(device)
#         for sparse_module in sparse_modules
#     ]
#     sparse_weights_spops = [
#         sparse_module.sparse_layer.to(device) for sparse_module in sparse_modules
#     ]

#     print("Testing provider:", provider, "K:", K)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == "lora":
#         ms, min_ms, max_ms = tn.testing.do_bench(
#             lambda: lora_merge(lora_a, lora_b, x, layer.weight.T, W_mege),
#             quantiles=quantiles,
#         )

#     # gbps = lambda ms: 2 * s * h * o * 2 * 1e-9 / (ms * 1e-3)
#     # return gbps(ms), gbps(max_ms), gbps(min_ms)
#     return ms, max_ms, min_ms


# benchmark.run(show_plots=True, print_data=True, save_path=".")
