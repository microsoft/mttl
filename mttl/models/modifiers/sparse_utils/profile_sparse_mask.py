import logging
import time

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
model_name = "EleutherAI/gpt-neo-125m"
keep_ratio = 0.002
block_size = 64
mask_updater = None
modify_layers = "q_proj|v_proj|k_proj"  # ,".*Wqkv.*|.*out_proj.*"


# Define input sizes and batch sizes for testing
max_seq_len = 1024
bs = 2
vocab_size = 32000

table = pd.DataFrame(
    columns=["Runtime", "Allocated Memory", "Reserved Memory", "Number of Parameters"]
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

    # Measure runtime
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(runs):
        loss = module(**input_data).loss
        loss.backward()
        module.zero_grad()
    torch.cuda.synchronize()
    end_time = time.time()
    avg_runtime = (end_time - start_time) / runs

    # Measure memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loss = module(**input_data).loss  # Forward pass to record memory
    loss.backward()  # Backward pass to record memory
    memory_allocated = torch.cuda.max_memory_allocated()
    memory_reserved = torch.cuda.max_memory_reserved()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return avg_runtime, memory_allocated, memory_reserved


############################################################################################################################################################
# Benchmarking LoRA

seed_everything(0)

adapter_config = LoRAConfig(modify_layers=modify_layers, lora_rank=8)
model = None
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


sparse_runtime, sparse_alloc, sparse_reserved = benchmark_module(model, runs=50)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"LoRA - Runtime: {sparse_runtime:.6f}s, Allocated Memory: {sparse_alloc / 1e6:.2f}MB, Reserved Memory: {sparse_reserved / 1e6:.2f}MB"
)
table.loc["LoRA"] = [sparse_runtime, sparse_alloc, sparse_reserved, n_params]

#################################################################################################################################################################
# Benchmarking BlcockSparseLinearModule

seed_everything(0)
adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="triton_block_sparse",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
model = None
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


# Run benchmarks
sparse_runtime, sparse_alloc, sparse_reserved = benchmark_module(model, runs=50)
print(
    f"BlockSparseLinearModule with block sparsity - Runtime: {sparse_runtime:.6f}s, Allocated Memory: {sparse_alloc / 1e6:.2f}MB, Reserved Memory: {sparse_reserved / 1e6:.2f}MB"
)
table.loc["BlockSparseLinearModule"] = [
    sparse_runtime,
    sparse_alloc,
    sparse_reserved,
    n_params,
]


#################################################################################################################################################################
# Benchmarking SparseLinearModule


seed_everything(0)
adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="sp_add+sp_mm",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
model = None
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


# Run benchmarks
sparse_runtime, sparse_alloc, sparse_reserved = benchmark_module(model, runs=50)
print(
    f"SparseLinearModule (spops) with regular sparsity - Runtime: {sparse_runtime:.6f}s, Allocated Memory: {sparse_alloc / 1e6:.2f}MB, Reserved Memory: {sparse_reserved / 1e6:.2f}MB"
)
table.loc["SparseLinearModule (reg sp.)"] = [
    sparse_runtime,
    sparse_alloc,
    sparse_reserved,
    n_params,
]

############################################################################################################################################################
# Benchmarking SparseLinearModule with block sparsity


seed_everything(0)
adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="sp_add+sp_mm",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
model = None
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

sparse_runtime, sparse_alloc, sparse_reserved = benchmark_module(model, runs=50)
print(
    f"SparseLinearModule (spops) with blcok sparsity - Runtime: {sparse_runtime:.6f}s, Allocated Memory: {sparse_alloc / 1e6:.2f}MB, Reserved Memory: {sparse_reserved / 1e6:.2f}MB"
)
table.loc["SparseLinearModule (block sp.)"] = [
    sparse_runtime,
    sparse_alloc,
    sparse_reserved,
    n_params,
]

#################################################################################################################################################################
#  Benchmarking MaskedLinear with regular sparsity

seed_everything(0)
adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="masked_linear",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
model = None
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

scattered_runtime, scattered_alloc, scattered_reserved = benchmark_module(
    model, runs=50
)
print(
    f"MaskedLinear with regular sparsity - Runtime: {scattered_runtime:.6f}s, Allocated Memory: {scattered_alloc / 1e6:.2f}MB, Reserved Memory: {scattered_reserved / 1e6:.2f}MB"
)
table.loc["MaskedLinear (reg. sp)"] = [
    scattered_runtime,
    scattered_alloc,
    scattered_reserved,
    n_params,
]

############################################################################################################################################################
# Benchmarking MaskedLinear with block sparsity

seed_everything(0)
adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="masked_linear",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
model = None
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

scattered_runtime, scattered_alloc, scattered_reserved = benchmark_module(
    model, runs=50
)
print(
    f"MaskedLinear with block sparsity - Runtime: {scattered_runtime:.6f}s, Allocated Memory: {scattered_alloc / 1e6:.2f}MB, Reserved Memory: {scattered_reserved / 1e6:.2f}MB"
)
table.loc["MaskedLinear (block sp.)"] = [
    scattered_runtime,
    scattered_alloc,
    scattered_reserved,
    n_params,
]

#################################################################################################################################################################
# Benchmarking ScatteredSparseLinearModule


seed_everything(0)
adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="scattered",
    sps_type="block_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
model = None
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

scattered_runtime, scattered_alloc, scattered_reserved = benchmark_module(
    model, runs=50
)
print(
    f"ScatteredSparseLinearModule with block sparsity - Runtime: {scattered_runtime:.6f}s, Allocated Memory: {scattered_alloc / 1e6:.2f}MB, Reserved Memory: {scattered_reserved / 1e6:.2f}MB"
)
table.loc["ScatteredSparseLinearModule (block sp.)"] = [
    scattered_runtime,
    scattered_alloc,
    scattered_reserved,
    n_params,
]

############################################################################################################################################################
# Benchmarking ScatteredSparseLinearModule with regular sparsity


seed_everything(0)
adapter_config = SparseMaskConfig(
    modify_layers=modify_layers,
    sps_impl="scattered",
    sps_type="regular_sparse",
    keep_ratio=keep_ratio,
    mask_updater=mask_updater,
    n_steps_in_mask_update=1,
    block_size=block_size,
)
model = None
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

scattered_runtime, scattered_alloc, scattered_reserved = benchmark_module(
    model, runs=50
)
print(
    f"ScatteredSparseLinearModule with regular sparsity - Runtime: {scattered_runtime:.6f}s, Allocated Memory: {scattered_alloc / 1e6:.2f}MB, Reserved Memory: {scattered_reserved / 1e6:.2f}MB"
)
table.loc["ScatteredSparseLinearModule (reg sp.)"] = [
    scattered_runtime,
    scattered_alloc,
    scattered_reserved,
    n_params,
]

print(table)
