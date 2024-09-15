import numpy as np
import stk
import stk.ops
import torch
import torch.nn.functional as F
import triton as tn
from spops import csr_add, sddmm
from triton.ops.blocksparse import matmul

from mttl.models.modifiers.sparse_mask import SparseMaskConfig, SparseWeights
from mttl.models.modifiers.sparse_utils.utils import init_sparse_weights

n_blocks = 8
BLOCK_SIZE = 128
dtype = torch.float16

sequence_length = 1024
hidden_size = 2048  # phi2 size
mlp_size = 8192
sparcity = n_blocks * (BLOCK_SIZE**2) / (hidden_size * mlp_size)
print(f"sparsity: {sparcity}")

# W = (
#     init_sparse_weights("block_sparse", sparcity, (hidden_size, mlp_size), BLOCK_SIZE)
#     .to("cuda")
#     .to(dtype)
#     .contiguous()
# )

# sparse_W = SparseWeights.from_dense(W, SparseMaskConfig(keep_ratio=sparcity, block_size=BLOCK_SIZE, sps_type="block_sparse")).to("cuda")

# idxs = torch.tensor(
#     np.array(sparse_W.twod_indices),
#     dtype=torch.int64,
#     device="cuda"
# )

def scatter_add(X_dense, values, idxs):
    row_indices, col_indices = idxs[0], idxs[1]
    flat_indices = row_indices * X_dense.size(1) + col_indices
    flat_weights = X_dense.view(-1)
    updated_flat_weights = flat_weights.scatter_add(0, flat_indices, values)

    weights = updated_flat_weights.view_as(X_dense)
    return weights
    

def stk_sdd(X, values, W):
    return csr_add(
            values, W.row_offs, W.row_idx, W.col_idx, X
        ) 


@tn.testing.perf_report(
    tn.testing.Benchmark(
        x_names=["h"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            128 * i for i in [8, 10, 12, 14, 16, 20]
        ],  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["scatter_add", "csr_add"],  # Possible values for `line_arg`.
        line_names=["scatter_add", "csr_add"],  # Possible values for `line_arg`.
        styles=[
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("red", "-"),
            ("purple", "-"),
            ("black", "-"),
        ],  # Line color and style.
        ylabel="ms",  #'GB/s',  # Label name for the y-axis.
        xlabel="seq length dim.",
        plot_name="matmul-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={
            "o": mlp_size,
            "sp": sparcity,
        },  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(h, o, sp, provider):
    X = torch.rand((h, o), device="cuda", dtype=dtype).contiguous()
    W = (
        init_sparse_weights("block_sparse", sp, (h, o), BLOCK_SIZE)
        .to("cuda")
        .to(dtype)
        .contiguous()
    )    
    sparse_W = SparseWeights.from_dense(W, SparseMaskConfig(keep_ratio=sparcity, block_size=BLOCK_SIZE, sps_type="block_sparse")).to("cuda")

    idxs = torch.tensor(
        np.array(sparse_W.twod_indices),
        dtype=torch.int64,
        device="cuda"
    )

    quantiles = [0.5, 0.2, 0.8]
    if provider == "scatter_add":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: scatter_add(X, sparse_W.sparse_weights.detach(), idxs), quantiles=quantiles
        )
    elif provider == "csr_add":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: stk_sdd(X, sparse_W.sparse_weights.detach(), sparse_W), quantiles=quantiles
        )
    # gbps = lambda ms: 2 * s * h * o * 2 * 1e-9 / (ms * 1e-3)
    # return gbps(ms), gbps(max_ms), gbps(min_ms)
    return ms, max_ms, min_ms


benchmark.run(show_plots=True, print_data=True, save_path=".")
