import logging
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from mttl.logging import logger
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.lora import LoRA, LoRAConfig
from mttl.models.modifiers.sparse_mask import (
    MaskedLinear,
    ScatteredSparseLinearModule,
    SparseLinearModule,
    SparseMaskAdapter,
    SparseMaskConfig,
)
from mttl.models.utils import model_loader_helper, transfer_batch_to_device

logger.setLevel(logging.ERROR)
model_name = "EleutherAI/gpt-neo-125m"  # "EleutherAI/gpt-neo-125m"  # "phi-2"
block_size = 64
n_blocks = 128
mask_updater = None
modify_layers = ".*q_proj.*|.*v_proj.*|.*k_proj.*"  # ".*q_proj.*|.*v_proj.*|.*k_proj.*"  # ".*Wqkv.*" #
n_iters = 50

in_d = 2048
out_d = 8192 *2
dtype=torch.bfloat16

# input sizes and batch sizes for testing
max_seq_len = 1024
bs = 5
vocab_size = 32000


def calculate_lora_parameters(input_dim, output_dim, rank):
    return input_dim * rank + output_dim * rank

layer = nn.Linear(in_d, out_d)
layer.weight.requires_grad_(False)
layer.bias.requires_grad_(False)

def find_hyperpaams():
    modules = {"linear": layer}
    modified_modules = {}
    keep_ratios = []
    lora_ranks = []

    for name, module in modules.items():
        keep_ratio = (
            n_blocks
            * (block_size**2)
            / (module.in_features * module.out_features)
        )
        tot_sparse_params = (
            module.in_features * module.out_features * keep_ratio
        )
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
print(f"Keep ratio: {keep_ratio}, LoRA rank: {lora_rank}")


table = pd.DataFrame(
    columns=[
        "Av. Runtime",
        "Av. Forward time",
        "Av. Backward time",
        "Allocated Memory",
        "Reserved Memory",
        "Number of Parameters",
    ]
)


def dummy_batch():
    torch.manual_seed(0)
    batch = {
        "input_ids": torch.randint(10, vocab_size, (bs, max_seq_len)),
        "labels": torch.randint(10, vocab_size, (bs, max_seq_len)),
    }
    seq_len = torch.randint(0, max_seq_len, (bs,))
    attn_mask = torch.zeros(bs, max_seq_len, dtype=torch.int32)
    attn_mask[torch.arange(bs), seq_len] = 1
    attn_mask = 1 - attn_mask.cumsum(dim=-1)
    batch["attention_mask"] = attn_mask
    return batch


def benchmark_module(module, runs=100):
    # Set up inputs
    input_data = dummy_batch()
    input_data = torch.rand(bs, max_seq_len, in_d).to("cuda").to(dtype)

    # Warm-up to ensure accurate measurement
    for _ in range(10):
        out = module(input_data)
        loss = torch.mean(out)
        loss.backward()
        module.zero_grad()

    forward_time_total = 0.0
    backward_time_total = 0.0

    # Benchmark runs
    for _ in range(runs):
        # Forward pass timing
        torch.cuda.synchronize()
        start_time = time.time()
        out = module(input_data)        
        loss = torch.mean(out)
        torch.cuda.synchronize()
        forward_time = time.time() - start_time

        # Backward pass timing
        torch.cuda.synchronize()
        start_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start_time

        # Zero gradients
        module.zero_grad()

        # Accumulate times
        forward_time_total += forward_time
        backward_time_total += backward_time

    avg_forward_time = forward_time_total / runs
    avg_backward_time = backward_time_total / runs
    avg_runtime = avg_forward_time + avg_backward_time

    # Measure memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = module(input_data)  # Forward pass to record memory
    loss = torch.mean(out)
    loss.backward()  # Backward pass to record memory
    memory_allocated = torch.cuda.max_memory_allocated()
    memory_reserved = torch.cuda.max_memory_reserved()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return (
        avg_runtime,
        avg_forward_time,
        avg_backward_time,
        memory_allocated,
        memory_reserved,
    )


def run_benchmark(name, adapter_config):
    seed_everything(0)   
    if isinstance(adapter_config, LoRAConfig):
        module = LoRA(adapter_config, layer)
    else:
        module = SparseMaskAdapter(adapter_config, layer)
    
    module.to("cuda").to(dtype)
    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

    runtime, forward_time, backward_time, sparse_alloc, sparse_reserved = (
        benchmark_module(module, runs=n_iters)
    )
    print(
        f"{name} - Runtime: {runtime:.6f}s, Allocated Memory: {sparse_alloc / 1e6:.2f}MB, Reserved Memory: {sparse_reserved / 1e6:.2f}MB, Number of Parameters: {n_params}"
    )
    table.loc[name] = [
        runtime,
        forward_time,
        backward_time,
        sparse_alloc,
        sparse_reserved,
        n_params,
    ]


############################################################################################################################################################
# Benchmarking LoRA

adapter_config = LoRAConfig(modify_layers=modify_layers, lora_rank=lora_rank)
run_benchmark("LoRA", adapter_config)

#################################################################################################################################################################
# Benchmarking BlcockSparseLinearModule + Dense without spoops

# adapter_config = SparseMaskConfig(
#     modify_layers=modify_layers,
#     sps_impl="dense+triton_block_sparse",
#     sps_type="block_sparse",
#     keep_ratio=keep_ratio,
#     reselection_steps=1,
#     block_size=block_size,
# )
# run_benchmark("BlockSparseLinearModule + Dense", adapter_config)

#################################################################################################################################################################
# Benchmarking BlcockSparseLinearModule without spoops

adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="triton_block_sparse_scatter",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)
run_benchmark("BlockSparseLinearModule (scatter add)", adapter_config)

#################################################################################################################################################################
# Benchmarking BlcockSparseLinearModule

adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="triton_block_sparse",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)
run_benchmark("BlockSparseLinearModule", adapter_config)


#################################################################################################################################################################
# Benchmarking SparseLinearModule


# adapter_config = SparseMaskConfig(
#     modify_layers=modify_layers,
#     sps_impl="sp_add+sp_mm",
#     sps_type="regular_sparse",
#     keep_ratio=keep_ratio,
#     # mask_updater=mask_updater,
#     reselection_steps=1,
#     block_size=block_size,
# )
# run_benchmark("SparseLinearModule (reg sp.)", adapter_config)

# ############################################################################################################################################################
# # Benchmarking SparseLinearModule with block sparsity


# adapter_config = SparseMaskConfig(
#     modify_layers=modify_layers,
#     sps_impl="sp_add+sp_mm",
#     sps_type="block_sparse",
#     keep_ratio=keep_ratio,
#     reselection_steps=1,
#     block_size=block_size,
# )
# run_benchmark("SparseLinearModule (block sp.)", adapter_config)

#################################################################################################################################################################
#  Benchmarking SPiEL with regular sparsity kernel


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="spiel",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)
run_benchmark("Spiel Linear (reg. sp)", adapter_config)



#################################################################################################################################################################
#  Benchmarking MaskedLinear with regular sparsity


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="masked_linear",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)
run_benchmark("MaskedLinear (reg. sp)", adapter_config)

############################################################################################################################################################
# Benchmarking MaskedLinear with block sparsity


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="masked_linear",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)
run_benchmark("MaskedLinear (block sp.)", adapter_config)

#################################################################################################################################################################
# Benchmarking ScatteredSparseLinearModule


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="scattered",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)
run_benchmark("ScatteredSparseLinearModule (block sp.)", adapter_config)

############################################################################################################################################################
# Benchmarking ScatteredSparseLinearModule with regular sparsity


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="scattered",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    reselection_steps=1,
    block_size=block_size,
)
run_benchmark("ScatteredSparseLinearModule (reg sp.)", adapter_config)

############################################################################################################################################################
# orer table by Av. Runtime
table = table.sort_values("Av. Runtime")
print(table)
# write table to a csv file
table.to_csv("benchmark_results.csv")
