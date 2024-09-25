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
from triton.ops.blocksparse import matmul

from mttl.logging import logger
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import Modifier
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRA, SkilledLoRAConfig
from mttl.models.modifiers.sparse_mask import (
    MaskedLinear,
    ScatteredSparseLinearModule,
    SparseLinearModule,
    SparseMaskAdapter,
    SparseMaskConfig,
)
from mttl.models.utils import model_loader_helper, transfer_batch_to_device

device = "cuda"
logger.setLevel(logging.ERROR)
block_size = 128  # 16
n_blocks = 16  # 1024

in_d = 2048
out_d = 8192
dtype = torch.bfloat16

# input sizes and batch sizes for testing
max_seq_len = 1024
bs = 5


layer = nn.Linear(in_d, out_d).to(device)
layer.weight.requires_grad_(False)
layer.bias.requires_grad_(False)
K = 10


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
x = torch.randn(bs, max_seq_len, in_d, dtype=dtype, device=device)


def create_adapter_set(adapter_config, layer, K) -> List[Modifier]:
    if isinstance(adapter_config, SparseMaskConfig):
        layer = nn.Linear(out_d, in_d)  # TODO: implement transpose in SparseWeights
        module = [SparseMaskAdapter(adapter_config, layer) for _ in range(K)]
    elif isinstance(adapter_config, LoRAConfig):
        module = [LoRA(adapter_config, layer) for _ in range(K)]
    return module


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


@torch.autocast(device_type="cuda", dtype=dtype)
def sparse_merge_and_forward(sparse_weights, x, W_base, W_merge):
    """
    Perform the merging of sparse adapters and compute the forward pass. This uses torch dds mm.

    Parameters:
    - sparse_weights: List[torch.Tensor], each of shape [input_dim, output_dim] in CSR format.
    - x: torch.Tensor, input of shape [bs, max_seq_len, input_dim].
    - W_base: torch.Tensor, base model weights of shape [input_dim, output_dim].
    - W_merge: torch.Tensor, merging weights of shape [bs, max_seq_len, K].

    Returns:
    - y: torch.Tensor, output of shape [bs, max_seq_len, output_dim].
    """
    bs, max_seq_len, input_dim = x.shape
    output_dim = W_base.shape[1]
    K = W_merge.shape[2]

    device = x.device
    dtype = x.dtype

    # Flatten x for efficient computation
    x_flat = x.reshape(bs * max_seq_len, input_dim)

    # Compute base output
    base_out = x_flat @ W_base

    # Initialize adapter output
    adapter_out = torch.zeros_like(base_out)

    # Iterate over each adapter
    for k in range(K):
        S_k = sparse_weights[k]  # Sparse matrix of shape [input_dim, output_dim]

        # Compute the output for this adapter
        output_k = (
            x_flat @ S_k
        )  # Shape: [bs * max_seq_len, output_dim] <- this is dds mm

        # Get the merging weights for this adapter
        W_k = W_merge[:, :, k].reshape(
            bs * max_seq_len, 1
        )  # Shape: [bs * max_seq_len, 1]

        # Scale and accumulate the adapter output
        adapter_out += output_k * W_k

    # Sum base and adapter outputs
    y_flat = base_out + adapter_out

    # Reshape back to [bs, max_seq_len, output_dim]
    y = y_flat.reshape(bs, max_seq_len, output_dim)

    return y


@torch.autocast(device_type="cuda", dtype=dtype)
def blck_sparse_merge_and_forward(sparse_weights, x, W_base, W_merge):
    bs, max_seq_len, input_dim = x.shape
    output_dim = W_base.shape[1]
    K = W_merge.shape[2]

    device = x.device
    dtype = x.dtype

    # Flatten x for efficient computation
    x_flat = x.reshape(bs * max_seq_len, input_dim)

    # Compute base output
    base_out = x_flat @ W_base

    # Initialize adapter output
    adapter_out = torch.zeros_like(base_out)

    # Iterate over each adapter
    for k in range(K):
        S_k = sparse_weights[k]  # Sparse matrix of shape [input_dim, output_dim]

        # Compute the output for this adapter
        output_k = F.linear(x_flat, S_k)  # Shape: [bs * max_seq_len, output_dim]

        # Get the merging weights for this adapter
        W_k = W_merge[:, :, k].reshape(
            bs * max_seq_len, 1
        )  # Shape: [bs * max_seq_len, 1]

        # Scale and accumulate the adapter output
        adapter_out += output_k * W_k

    # Sum base and adapter outputs
    y_flat = base_out + adapter_out

    # Reshape back to [bs, max_seq_len, output_dim]
    y = y_flat.reshape(bs, max_seq_len, output_dim)

    return y


@torch.autocast(device_type="cuda", dtype=dtype)
def sparse_merge_and_forward_with_SpMM(sparse_weights, x, W_base, W_merge):
    bs, max_seq_len, input_dim = x.shape
    output_dim = W_base.shape[1]
    K = W_merge.shape[2]

    device = x.device
    dtype = x.dtype

    # Flatten x for efficient computation
    x_flat = x.reshape(bs * max_seq_len, input_dim)

    # Compute base output
    base_out = x_flat @ W_base

    # Initialize adapter output
    adapter_out = torch.zeros_like(base_out)

    # Iterate over each adapter
    for k in range(K):
        S_k = sparse_weights[k]  # Sparse matrix of shape [input_dim, output_dim]

        # Compute the output for this adapter
        output_k = spmm(
            S_k.sparse_weights,
            S_k.row_offs,
            S_k.row_idx,
            S_k.col_idx,
            x_flat.T.contiguous(),
            S_k.shape[0],
            backend="sputnik",
        )  # Shape: [bs * max_seq_len, output_dim]

        # Get the merging weights for this adapter
        W_k = W_merge[:, :, k].reshape(
            bs * max_seq_len, 1
        )  # Shape: [bs * max_seq_len, 1]

        # Scale and accumulate the adapter output
        adapter_out += output_k.T * W_k

    # Sum base and adapter outputs
    y_flat = base_out + adapter_out

    # Reshape back to [bs, max_seq_len, output_dim]
    y = y_flat.reshape(bs, max_seq_len, output_dim)

    return y


@torch.autocast(device_type="cuda", dtype=dtype)
def sparse_merge_and_forward_with_spadd(sparse_weights, x, W_base, W_merge):
    bs, max_seq_len, input_dim = x.shape
    output_dim = W_base.shape[1]
    K = W_merge.shape[2]

    device = x.device
    dtype = x.dtype

    # Flatten x for efficient computation
    x_flat = x.reshape(bs * max_seq_len, input_dim)

    # Compute base output
    base_out = x_flat @ W_base

    # Initialize adapter output
    adapter_w = torch.zeros(sparse_weights[0].shape).to(device)

    # Iterate over each adapter
    for k in range(K):
        W_k = W_merge[:, :, k].reshape(
            bs * max_seq_len, 1
        )  # Shape: [bs * max_seq_len, 1]
        S_k = sparse_weights[k]

        # Compute the output for this adapter
        adapter_w = csr_add(
            S_k.sparse_weights * W_k, S_k.row_offs, S_k.row_idx, S_k.col_idx, adapter_w
        )

    adapter_out = x_flat @ adapter_w
    # Sum base and adapter outputs
    y_flat = base_out + adapter_out

    # Reshape back to [bs, max_seq_len, output_dim]
    y = y_flat.reshape(bs, max_seq_len, output_dim)

    return y


@torch.autocast(device_type="cuda", dtype=dtype)
def sparse_merge_and_forward_vectorized(sparse_weights, x, W_base, W_merge):
    """
    Perform the merging of sparse adapters and compute the forward pass.

    Parameters:
    - sparse_weights: List[torch.Tensor], each of shape [input_dim, output_dim] in CSR format.
    - x: torch.Tensor, input of shape [bs, max_seq_len, input_dim].
    - W_base: torch.Tensor, base model weights of shape [input_dim, output_dim].
    - W_merge: torch.Tensor, merging weights of shape [bs, max_seq_len, K].

    Returns:
    - y: torch.Tensor, output of shape [bs, max_seq_len, output_dim].
    """
    bs, max_seq_len, input_dim = x.shape
    output_dim = W_base.shape[1]
    K = W_merge.shape[2]

    device = x.device
    dtype = x.dtype

    # Flatten x for efficient computation
    x_flat = x.reshape(bs * max_seq_len, input_dim)

    # Compute base output
    base_out = x_flat @ W_base

    # Stack and expand sparse weights
    # Convert sparse weights to dense (if memory allows)
    sparse_weights_dense = torch.stack(
        [S.to_dense() for S in sparse_weights], dim=0
    )  # [K, input_dim, output_dim]

    # Compute adapter outputs
    # [bs*max_seq_len, K, output_dim]
    adapter_out = torch.einsum("bi,kio->bko", x_flat, sparse_weights_dense)
    W_merge_flat = W_merge.reshape(bs * max_seq_len, K, 1)  # [bs*max_seq_len, K, 1]
    adapter_out = (adapter_out * W_merge_flat).sum(
        dim=1
    )  # [bs*max_seq_len, output_dim]

    # Sum base and adapter outputs
    y_flat = base_out + adapter_out

    # Reshape back to [bs, max_seq_len, output_dim]
    y = y_flat.reshape(bs, max_seq_len, output_dim)

    return y


@torch.autocast(device_type="cuda", dtype=dtype)
def blk_sparse_merge_and_forward_triton(
    block_sparse_ops, block_sparse_weights, x, W_base, W_merge
):
    """
    Perform the merging of sparse adapters and compute the forward pass. This uses triton dds kernel with precomputed layour (see prepare_triton_bs_op).

    Parameters:
    - block_sparse_ops: List[triton.ops.blocksparse.matmul], each of shape [input_dim, output_dim] in CSR format.
    - block_sparse_weights: List[torch.Tensor], each of shape [input_dim, output_dim] in BSR format (these are only non-zero blocks).
    - x: torch.Tensor, input of shape [bs, max_seq_len, input_dim].
    - W_base: torch.Tensor, base model weights of shape [input_dim, output_dim].
    - W_merge: torch.Tensor, merging weights of shape [bs, max_seq_len, K].

    Returns:
    - y: torch.Tensor, output of shape [bs, max_seq_len, output_dim].
    """
    bs, max_seq_len, input_dim = x.shape
    output_dim = W_base.shape[1]
    K = W_merge.shape[2]

    device = x.device
    dtype = x.dtype

    # Flatten x for efficient computation
    x_flat = x.reshape(bs * max_seq_len, input_dim)

    # Compute base output
    base_out = x_flat @ W_base

    # Initialize adapter output
    adapter_out = torch.zeros_like(base_out)

    # Iterate over each adapter
    for k in range(K):
        S_k = block_sparse_weights[k]  # Sparse matrix of shape [input_dim, output_dim]
        _x_flat = x_flat.unsqueeze(0).unsqueeze(0).contiguous()
        # Compute the output for this adapter
        output_k = block_sparse_ops[k](
            _x_flat, S_k
        ).squeeze()  # Shape: [bs * max_seq_len, output_dim]

        # Get the merging weights for this adapter
        W_k = W_merge[:, :, k].reshape(
            bs * max_seq_len, 1
        )  # Shape: [bs * max_seq_len, 1]

        # Scale and accumulate the adapter output
        adapter_out += output_k * W_k

    # Sum base and adapter outputs
    y_flat = base_out + adapter_out

    # Reshape back to [bs, max_seq_len, output_dim]
    y = y_flat.reshape(bs, max_seq_len, output_dim)

    return y


@torch.autocast(device_type="cuda", dtype=dtype)
def blk_sparse_merge_and_forward_stk(block_sparse_weights, x, W_base, W_merge):
    bs, max_seq_len, input_dim = x.shape
    output_dim = W_base.shape[1]
    K = W_merge.shape[2]

    device = x.device
    dtype = x.dtype

    # Flatten x for efficient computation
    x_flat = x.reshape(bs * max_seq_len, input_dim)

    # Compute base output
    base_out = x_flat @ W_base

    # Initialize adapter output
    adapter_out = torch.zeros_like(base_out)

    # Iterate over each adapter
    for k in range(K):
        S_k = block_sparse_weights[k]  # Sparse matrix of shape [input_dim, output_dim]
        # Compute the output for this adapter
        output_k = stk.ops.dds(x_flat, S_k)  # Shape: [bs * max_seq_len, output_dim]

        # Get the merging weights for this adapter
        W_k = W_merge[:, :, k].reshape(
            bs * max_seq_len, 1
        )  # Shape: [bs * max_seq_len, 1]

        # Scale and accumulate the adapter output
        adapter_out += output_k * W_k

    # Sum base and adapter outputs
    y_flat = base_out + adapter_out

    # Reshape back to [bs, max_seq_len, output_dim]
    y = y_flat.reshape(bs, max_seq_len, output_dim)

    return y


sparse_merge_and_forward_compiled = torch.compile(sparse_merge_and_forward)
lora_merge_compiled = torch.compile(lora_merge)

# adapter_config_sm = SparseMaskConfig(
#     sps_impl="scattered",
#     sps_type="regular_sparse",
#     keep_ratio=keep_ratio,
#     reselection_steps=1,
#     block_size=block_size,
# )


adapter_config_lora = LoRAConfig(modify_layers="", lora_rank=lora_rank)
adapter_config_bs = SparseMaskConfig(
    sps_impl="scattered",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)


def bsr_to_binary_layout(bsr_matrix, block_size):
    # Get the shape of the BSR matrix
    M, K = bsr_matrix.shape

    # Number of blocks along rows and columns
    num_block_rows = M // block_size
    num_block_cols = K // block_size

    # Initialize the binary layout matrix with zeros
    binary_layout = torch.zeros((num_block_rows, num_block_cols), dtype=int)

    # Get BSR matrix data
    block_row_indices = bsr_matrix.col_indices()
    block_row_pointers = bsr_matrix.crow_indices()

    # Iterate over the block rows
    for block_row in range(num_block_rows):
        # Iterate over the non-zero blocks in the current block row
        for idx in range(
            block_row_pointers[block_row], block_row_pointers[block_row + 1]
        ):
            block_col = block_row_indices[idx]
            # Mark the block as non-zero
            binary_layout[block_row, block_col] = 1

    return binary_layout


def prepare_triton_bs_op(W, op_mode):
    Z, H = 1, 1
    AT = False
    BT = False

    layout = bsr_to_binary_layout(W, block_size).unsqueeze(0)
    # creat inputs
    op = matmul(layout, block_size, op_mode, trans_a=AT, trans_b=BT, device="cuda")
    return op


@tn.testing.perf_report(
    tn.testing.Benchmark(
        x_names=["K"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2, 3, 4],  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "stk",
            "triton_blck_sparse",
            "lora",
            "torch_sparse",
            "torch_block_sparse",
        ],  # "lora_compiled", "torch_sparse_compiled"],  # Possible values for `line_arg`.
        line_names=[
            "stk",
            "triton_blck_sparse",
            "lora",
            "torch_sparse",
            "torch_block_sparse",
        ],  # "lora_compiled", "torch_sparse_compiled"],  # Label name for the lines.
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("red", "-"),
            ("purple", "-"),
            ("black", "-"),
            ("brown", "-"),
        ],  # Line color and style.
        ylabel="ms",  #'GB/s',  # Label name for the y-axis.
        xlabel="K",
        plot_name="matmul-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={"bs": bs, "max_seq_len": max_seq_len, "in_d": in_d, "d_out": out_d},
    )
)
def benchmark(K, bs, max_seq_len, in_d, d_out, provider):
    W_mege = torch.randn(bs, max_seq_len, K, dtype=dtype, device=device)
    loras = create_adapter_set(adapter_config_lora, layer, K)
    sparse_modules = create_adapter_set(adapter_config_bs, layer, K)
    W_mege = W_mege.to(dtype=loras[0].lora_a.dtype)

    lora_a = torch.stack([lora.lora_a for lora in loras], dim=0)
    lora_b = torch.stack([lora.lora_b for lora in loras], dim=0)
    sparse_weights: List[torch.Tensor] = [
        sparse_module.sparse_layer.to_dense().to_sparse_csr().to(device)
        for sparse_module in sparse_modules
    ]
    sparse_weights_spops = [
        sparse_module.sparse_layer.to(device) for sparse_module in sparse_modules
    ]

    print("Testing provider:", provider, "K:", K)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "lora":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: lora_merge(lora_a, lora_b, x, layer.weight.T, W_mege),
            quantiles=quantiles,
        )
    elif provider == "lora_compiled":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: lora_merge_compiled(lora_a, lora_b, x, layer.weight.T, W_mege),
            quantiles=quantiles,
        )
    elif provider == "torch_sparse":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: sparse_merge_and_forward(sparse_weights, x, layer.weight.T, W_mege),
            quantiles=quantiles,
        )
    elif provider == "torch_block_sparse":
        block_sparse_weights: List[torch.Tensor] = [
            sparse_module.sparse_layer.to_dense().T.to_sparse_bsr(block_size).to(device)
            for sparse_module in sparse_modules
        ]
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: blck_sparse_merge_and_forward(
                block_sparse_weights, x, layer.weight.T, W_mege
            ),
            quantiles=quantiles,
        )
    elif provider == "torch_sparse_compiled":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: sparse_merge_and_forward_compiled(
                sparse_weights, x, layer.weight.T, W_mege
            ),
            quantiles=quantiles,
        )
    elif provider == "sparse_vectorized":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: sparse_merge_and_forward_vectorized(
                sparse_weights, x, layer.weight.T, W_mege
            ),
            quantiles=quantiles,
        )
    elif provider == "sparse_spadd":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: sparse_merge_and_forward_with_spadd(
                sparse_weights_spops, x, layer.weight.T, W_mege
            ),
            quantiles=quantiles,
        )
    elif provider == "triton_blck_sparse":
        block_sparse_weights: List[torch.Tensor] = [
            sparse_module.sparse_layer.to_dense().to_sparse_bsr(block_size).to(device)
            for sparse_module in sparse_modules
        ]
        # create a list of ops with precomputed layouts for the BSR matrices
        block_sparse_ops = [
            prepare_triton_bs_op(sparse_w, "dds") for sparse_w in block_sparse_weights
        ]
        # block_sparse_weights_as_dense = [
        #     sparse_w.to_dense()
        #     .to(dtype)
        #     .reshape(-1, block_size, block_size)
        #     .unsqueeze(0)
        #     .contiguous()
        #     for sparse_w in block_sparse_weights
        # ]
        block_sparse_weights = [
            sparse_w.values().to(dtype).unsqueeze(0).contiguous()
            for sparse_w in block_sparse_weights
        ]

        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: blk_sparse_merge_and_forward_triton(
                block_sparse_ops,
                block_sparse_weights,
                x,
                layer.weight.T,
                W_mege,
            ),
            quantiles=quantiles,
        )

    elif provider == "stk":
        # only supports block_size = 128 and float16
        if block_size != 128:
            ms, min_ms, max_ms = 0, 0, 0
        else:
            block_sparse_weights = []
            for sparse_module in sparse_modules:
                W = sparse_module.sparse_layer.to_dense().to(device).to(torch.float16)
                W_stk = stk.ops.to_sparse(W, blocking=block_size)
                W_stk.validate()
                block_sparse_weights.append(W_stk)
            ms, min_ms, max_ms = tn.testing.do_bench(
                lambda: blk_sparse_merge_and_forward_stk(
                    block_sparse_weights, x, layer.weight.T, W_mege
                ),
                quantiles=quantiles,
            )

    # gbps = lambda ms: 2 * s * h * o * 2 * 1e-9 / (ms * 1e-3)
    # return gbps(ms), gbps(max_ms), gbps(min_ms)
    return ms, max_ms, min_ms


benchmark.run(show_plots=True, print_data=True, save_path=".")
