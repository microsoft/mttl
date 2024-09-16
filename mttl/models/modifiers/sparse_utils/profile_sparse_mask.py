import logging
import re
import time

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything

from mttl.logging import logger
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.lora import LoRAConfig
from mttl.models.modifiers.sparse_mask import (
    MaskedLinear,
    ScatteredSparseLinearModule,
    SparseLinearModule,
    SparseMaskConfig,
)
from mttl.models.utils import model_loader_helper, transfer_batch_to_device

logger.setLevel(logging.ERROR)
model_name = "phi-2"  # "EleutherAI/gpt-neo-125m"  # "phi-2"
block_size = 128
n_blocks = 6
mask_updater = None
modify_layers = ".*Wqkv.*"  # ".*q_proj.*|.*v_proj.*|.*k_proj.*"  # ".*Wqkv.*" #
n_iters = 50

# input sizes and batch sizes for testing
max_seq_len = 1024
bs = 1
vocab_size = 32000


def calculate_lora_parameters(input_dim, output_dim, rank):
    return input_dim * rank + output_dim * rank


def find_hyperpaams():

    model = model_loader_helper(
        model_name,
        bf16=True,
        fp16=False,
        load_in_4bit=False,
        load_in_8bit=False,
        device_map="cpu",
    )
    modules = dict(model.named_modules())
    modified_modules = {}
    keep_ratios = []
    lora_ranks = []

    for ml in modify_layers.split("|"):
        for name, module in modules.items():
            if re.match(ml, name) and ml not in modified_modules:
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
                modified_modules[ml] = {
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
    input_data = transfer_batch_to_device(input_data, "cuda")

    # Warm-up to ensure accurate measurement
    for _ in range(10):
        loss = module(**input_data).loss
        loss.backward()
        module.zero_grad()

    forward_time_total = 0.0
    backward_time_total = 0.0

    # Benchmark runs
    for _ in range(runs):
        # Forward pass timing
        torch.cuda.synchronize()
        start_time = time.time()
        loss = module(**input_data).loss
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
    loss = module(**input_data).loss  # Forward pass to record memory
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
    model = model_loader_helper(
        model_name,
        bf16=True,
        fp16=False,
        load_in_4bit=False,
        load_in_8bit=False,
        device_map="cpu",
    )
    modify_transformer(model, adapter_config)
    model.to("cuda")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    runtime, forward_time, backward_time, sparse_alloc, sparse_reserved = (
        benchmark_module(model, runs=n_iters)
    )
    print(
        f"{name} - Runtime: {runtime:.6f}s, Allocated Memory: {sparse_alloc / 1e6:.2f}MB, Reserved Memory: {sparse_reserved / 1e6:.2f}MB"
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
# Benchmarking BlcockSparseLinearModule

adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="triton_block_sparse",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
run_benchmark("BlockSparseLinearModule", adapter_config)


#################################################################################################################################################################
# Benchmarking SparseLinearModule


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="sp_add+sp_mm",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
run_benchmark("SparseLinearModule (reg sp.)", adapter_config)

############################################################################################################################################################
# Benchmarking SparseLinearModule with block sparsity


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="sp_add+sp_mm",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
run_benchmark("SparseLinearModule (block sp.)", adapter_config)

#################################################################################################################################################################
#  Benchmarking MaskedLinear with regular sparsity


adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="masked_linear",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
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
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
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
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
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
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
run_benchmark("ScatteredSparseLinearModule (reg sp.)", adapter_config)

############################################################################################################################################################
print(table)
