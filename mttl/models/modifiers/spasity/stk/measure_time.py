import time
from functools import partial

import numpy as np
import stk
import torch
import torch.nn.functional as F
from absl.testing import parameterized
from pytorch_lightning import seed_everything
from stk.matrix import Matrix

from mttl.models.modifiers.spasity.stk import functions, linear_ops, matrix_ops
from mttl.models.modifiers.spasity.stk.linear_ops_test_scatter import (
    _dense_and_sparse,
    dumb_forward,
)


def benchmark_module(name, function, runs=100):
    # Warm-up to ensure accurate measurement
    for _ in range(10):
        out = function()

    forward_time_total = 0.0

    # Benchmark runs
    for _ in range(runs):
        # Forward pass timing
        torch.cuda.synchronize()
        start_time = time.time()
        out = function()
        torch.cuda.synchronize()
        forward_time = time.time() - start_time

        # Accumulate times
        forward_time_total += forward_time

    avg_forward_time = forward_time_total / runs

    # Measure memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = function()  # Forward pass to record memory
    memory_allocated = torch.cuda.max_memory_allocated()
    memory_reserved = torch.cuda.max_memory_reserved()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        f"Average forward time {name}: {avg_forward_time:.6f}s",
        f"Memory allocated: {memory_allocated/1024**2:.2f}MB",
        f"Memory reserved: {memory_reserved/1024**2:.2f}MB",
    )


def calculate_lora_parameters(input_dim, output_dim, rank):
    return input_dim * rank + output_dim * rank


def find_lora_hyperpaams(d_in, d_out, tot_sparse_params):
    lora_ranks = []
    lora_rank = 1
    for rank in range(1, d_in):
        lora_params = calculate_lora_parameters(d_in, d_out, rank)
        if lora_params <= tot_sparse_params:
            lora_rank = rank
        else:
            break
    lora_ranks.append(lora_rank)
    return int(np.mean(lora_ranks))


SC_MOE_TEST = {
    # bs, d, h, E, k, sparsity, blocking, dtype
    (1024, 2048, 8192, 20, 2, 0.995, 16, torch.float16),
    (1024, 2048, 8192, 20, 2, 0.9, 128, torch.float16),
    (1024, 2048, 8192, 100, 2, 0.995, 16, torch.float16),
    (1024, 2048, 8192, 100, 2, 0.9, 128, torch.float16),
    # (1024, 1024, 8192, 20, 2, 0.8, 16, torch.float16),
    # (1024, 1024, 2048, 20, 2, 0.8, 16, torch.float16),
    # (8, 128, 256, 10, 2, 0.8, 16, torch.float16),
}


for bs, d, h, E, k, sparsity, blocking, dtype in SC_MOE_TEST:
    print("=====================================================================")
    print(
        f"*****   Running test with bs={bs}, d={d}, h={h}, E={E}, k={k}, sparsity={sparsity}, blocking={blocking}, dtype={dtype}  *****"
    )

    torch.manual_seed(42)
    # print(f"Running test with bs={bs}, d={d}, h={h}, E={E}, k={k}, sparsity={sparsity}, blocking={blocking}, dtype={dtype}")
    logits = torch.randn(bs, E, dtype=dtype)
    weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
    X = torch.randn(bs, d, dtype=dtype, requires_grad=True).cuda()
    W = torch.randn(d, h, dtype=dtype, requires_grad=True).cuda()
    adaps = [_dense_and_sparse(d, h, sparsity, blocking, dtype) for _ in range(E)]
    adaps_sparse = [adap[1] for adap in adaps]
    adaps_dense = [adap[0] for adap in adaps]
    ada_data = torch.stack([adap.data for adap in adaps_sparse], dim=0)
    row_idxs = torch.stack([adap.row_indices for adap in adaps_sparse], dim=0)
    col_idxs_t = torch.stack([adap.column_indices_t for adap in adaps_sparse], dim=0)
    offsets_t = torch.stack([adap.offsets_t for adap in adaps_sparse], dim=0)
    block_offsets_t = torch.stack(
        [adap.block_offsets_t for adap in adaps_sparse], dim=0
    )

    k_weights, expert_idxs = torch.topk(weights, k)

    def call_with_baseact_and_idxs_computation(X, W, expert_idxs, function, **kwargs):
        base_act = torch.matmul(X, W)
        sorted_expert_idxs, sorted_scattered_idxs = linear_ops.flatten_and_sort(
            expert_idxs
        )
        padded_block_idxs, expert_offsets = linear_ops.padded_block_indices(
            sorted_expert_idxs, E
        )
        return function(
            x=X,
            base_act=base_act,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            **kwargs,
        )

    # base_act = torch.matmul(X, W)
    func = partial(
        call_with_baseact_and_idxs_computation,
        X=X,
        W=W,
        expert_idxs=expert_idxs,
        function=linear_ops.scattergather_adamerge,
        k=k,
        ada_weights=ada_data,
        row_idxs=row_idxs,
        col_idxs=col_idxs_t,
        offsets=offsets_t,
        block_offsets_t=block_offsets_t,
        ada_block_size=blocking,
        gates=k_weights,
    )
    benchmark_module("BS kernel not optimized", func)
    # func_dummb = partial(dumb_forward, base_act=base_act, x=X, expert_p=k_weights, expert_idxs=expert_idxs, adaps=adaps_dense)
    # benchmark_module("dummy forward", func_dummb)

    func_opt = partial(
        call_with_baseact_and_idxs_computation,
        X=X,
        W=W,
        expert_idxs=expert_idxs,
        function=linear_ops.scattergather_adamerge2,
        k=k,
        ada_weights=ada_data,
        row_idxs=row_idxs,
        col_idxs=col_idxs_t,
        offsets=offsets_t,
        block_offsets_t=block_offsets_t,
        ada_block_size=blocking,
        gates=k_weights,
    )
    benchmark_module("BS kernel optimized", func)
    lora_rank = find_lora_hyperpaams(d, h, np.prod(ada_data.shape[1:]))

    def lora_merge(lora_a, lora_b, x, W_base, W_merge):
        # LoRA does not profit from lower top-k in this vanila form
        # merge into 1 lora
        A = torch.einsum("be,edr->bdr", (W_merge, lora_a))
        B = torch.einsum("be,erd->brd", (W_merge, lora_b))
        # lora forward
        partial_out = torch.einsum("bd,bdr->br", (x, A))
        adapter_out = torch.einsum("br,brd->bd", (partial_out, B))
        dense_out = x @ W_base
        return adapter_out + dense_out

    lora_a = torch.randn(E, d, lora_rank, dtype=dtype).cuda().contiguous()
    lora_b = torch.randn(E, lora_rank, h, dtype=dtype).cuda().contiguous()
    func_lora = partial(
        lora_merge, lora_a=lora_a, lora_b=lora_b, x=X, W_base=W, W_merge=weights
    )
    benchmark_module("LoRA merge (our current vanila)", func_lora)
