from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

def compile_extension():
    cuda_source = Path("/home/mila/o/ostapeno/dev/mttl_public/mttl/models/modifiers/sparse_utils/sparse_merge/merge.cu").read_text()
    cpp_source =  "torch::Tensor bmm_w_merge(torch::Tensor A, torch::Tensor B, torch::Tensor W, torch::Tensor SPA);"


    # Load the CUDA kernel as a PyTorch extension
    ext = load_inline(
        name="bmm_w_merge_wmma_module",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["bmm_w_merge"],
        with_cuda=True,
        extra_cuda_cflags=[
            '-gencode', 'arch=compute_80,code=sm_80',  # Adjust for A100 GPU
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda', "-DTORCH_USE_CUDA_DSA"
        ],
        build_directory='/home/mila/o/ostapeno/dev/mttl_public/mttl/models/modifiers/sparse_utils/sparse_merge/build',
    )
    return ext

# Load the compiled module
batched_matmul_wmma_merge_module = compile_extension()

def batched_merge_matmul(A, B, W, Adapters):
    assert A.dtype == torch.half and B.dtype == torch.half
    assert A.is_cuda and B.is_cuda

    batch_size, M, K = A.shape
    _, Kb, N = B.shape
    assert K == Kb
    res = batched_matmul_wmma_merge_module.bmm_w_merge(A, B, W, Adapters)
    return res

def manual_merge_matmul(A, B, W, Adapters):
    merged_adapters = torch.einsum("bk,knm->bnm", W, Adapters)
    out = torch.bmm(A, B  + merged_adapters)
    return out


# Set the seed for reproducibility
torch.manual_seed(0)
# Define matrix dimensions
batch_size = 128
M = 256
K = 256
N = 256
E = 10

# Create random half-precision tensors
A = torch.randn(batch_size, M, K, device='cuda', dtype=torch.half).contiguous()
B = torch.randn(batch_size, K, N, device='cuda', dtype=torch.half).contiguous()
#random between 0 and 1
W = torch.randn(batch_size, E, device='cuda', dtype=torch.half).contiguous()
Adapters = torch.randn(E, K, N, device='cuda', dtype=torch.half).contiguous()

# batched_matmul_wmma_python(A, B)

# Warm-up
for _ in range(1):
    C_custom = batched_merge_matmul(A, B, W, Adapters)

# Measure performance
import time

start = time.time()  
C_custom = batched_merge_matmul(A, B, W, Adapters)
torch.cuda.synchronize()
end = time.time()
print(f"Custom kernel time: {end - start:.6f} seconds")

start = time.time()
C_reference = manual_merge_matmul(A, B, W, Adapters)
torch.cuda.synchronize()
end = time.time()
print(f"PyTorch bmm time: {end - start:.6f} seconds")

C_custom = C_custom.to(C_reference.dtype)
# Verify correctness
max_error = (C_custom - C_reference).abs().max().item()
print(f"Max error: {max_error}")
# C_reference = C_reference.to(C_custom.dtype)
# C_custom = C_custom.to(C_reference.dtype)
print(torch.allclose(C_custom, C_reference, atol=3e-1))