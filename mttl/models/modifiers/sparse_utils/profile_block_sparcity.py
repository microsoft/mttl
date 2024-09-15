# several options to compare for block sparce operations:
# 1. triton.ops.blocksparse -- this is supposed to work fast for cases whee the sparcity structure is not changing too often
# 2. stk -- https://github.com/stanford-futuredata/stk -- this is supposed to work fast for cases where the sparcity structure is changing often
import stk
import stk.ops
import torch
import torch.nn.functional as F
import triton as tn
from spops import csr_add, sddmm
from triton.ops.blocksparse import matmul

from mttl.models.modifiers.sparse_mask import SparseMaskConfig, SparseWeights
from mttl.models.modifiers.sparse_utils.utils import init_sparse_weights

n_blocks = 4
BLOCK_SIZE = 128
dtype = torch.bfloat16

sequence_length = 1024
hidden_size = 2048  # phi2 size
mlp_size = 8192

sparcity = n_blocks * (BLOCK_SIZE**2) / (hidden_size * mlp_size)
print(f"sparsity: {sparcity}")

# W = init_sparse_weights("block_sparse", 0.005, (K, N), BLOCK_SIZE).contiguous().to('cuda')
# X = torch.randn(M, K).to('cuda').contiguous()


def stk_sdd(X, W, topo):
    return stk.ops.sdd(X, W, topo)


def torch_linear(X, W):
    return F.linear(X, W)


def spops_sdd_structure_aware(X, W, topo: SparseWeights):
    return sddmm(topo.row_offs, topo.row_idx, topo.col_idx, X, W)


def spops_sdd_sputnik(X, W, topo: SparseWeights):
    return sddmm(topo.row_offs, topo.row_idx, topo.col_idx, X, W, backend="sputnik")


def torch_linear_w_sparse(X, W):
    return F.linear(X, W)


def triton_blocksparse_mm(X, W, op):
    return op(X, W)


def prepare_triton_bs_op(X, W):
    Z, H = 1, 1
    AT = False
    BT = False
    op_mode = "sdd"

    def to_block_sparse_layout(matrix: torch.Tensor, block_size: int):
        """
        Returns layout of block sparse matrix: i.e. a matrix of shape (M//block_size, N//block_size) where each element is a boolean indicating whether the block is non-zero.
        """
        M, N = matrix.shape
        assert M % block_size == 0, "M must be divisible by block_size"
        assert N % block_size == 0, "N must be divisible by block_size"
        matrix = matrix.reshape(
            M // block_size,
            block_size,
            N // block_size,
            block_size,
        ).permute(0, 2, 1, 3)
        matrix = matrix.flatten(2, 3).sum(dim=-1)
        return matrix.cpu().bool().to(torch.int64)

    layout = to_block_sparse_layout(W, BLOCK_SIZE).unsqueeze(0)
    # creat inputs
    op = matmul(layout, BLOCK_SIZE, op_mode, trans_a=AT, trans_b=BT, device="cuda")
    return op


# # adapted from https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
@tn.testing.perf_report(
    tn.testing.Benchmark(
        # x_names=['o'],  # Argument names to use as an x-axis for the plot.
        # x_vals=[128*i for i in [8, 10, 20, 50, 64, 100]],  # Different possible values for `x_name`.
        x_names=["s"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            128 * i for i in [8, 10, 12, 14, 16, 20]
        ],  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["naive", "stk", "triton_bs"],  # Possible values for `line_arg`.
        line_names=["Naive", "stk", "triton_bs"],  # Label name for the lines.
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
            "h": hidden_size,
            "o": mlp_size,
            "sp": sparcity,
        },  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(s, h, o, sp, provider):
    X = torch.rand((s, h), device="cuda", dtype=dtype).contiguous()
    W = (
        init_sparse_weights("block_sparse", sp, (h, o), BLOCK_SIZE)
        .to("cuda")
        .to(dtype)
        .contiguous()
    )
    W_row_sparse = (
        init_sparse_weights("row_sparse", sp, (h, o), BLOCK_SIZE)
        .to("cuda")
        .to(dtype)
        .contiguous()
    )
    WT = W.T
    assert W.sum() > 0
    assert W_row_sparse.sum() == W.sum()


    quantiles = [0.5, 0.2, 0.8]
    if provider == "naive":
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: torch_linear(X, WT), quantiles=quantiles
        )
    if provider == "stk":      
        if BLOCK_SIZE != 128 or dtype != torch.float16:
            ms, min_ms, max_ms = 0, 0, 0
        else:                      
            W_stk = stk.ops.to_sparse(W, blocking=BLOCK_SIZE)
            W_stk.validate()
            ms, min_ms, max_ms = tn.testing.do_bench(
                lambda: stk_sdd(X, W, W_stk), quantiles=quantiles
            )
    if provider == "torch_bsr":
        W_bst = WT.to_sparse_bsr(blocksize=BLOCK_SIZE)
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: torch_linear(X, W_bst), quantiles=quantiles
        )
    if provider == "spops_block":
        W_spops_block = SparseWeights.from_dense(
            W,
            SparseMaskConfig(
                keep_ratio=sp, block_size=BLOCK_SIZE, sps_type="block_sparse"
            ),
        ).to("cuda")
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: spops_sdd_structure_aware(X, W, W_spops_block), quantiles=quantiles
        )
    if provider == "spops_row":
        W_row_sparse = (
            init_sparse_weights("row_sparse", sp, (h, o), BLOCK_SIZE)
            .to("cuda")
            .to(dtype)
            .contiguous()
        )
        W_spops_row = SparseWeights.from_dense(
            W_row_sparse,
            SparseMaskConfig(
                keep_ratio=sp, block_size=BLOCK_SIZE, sps_type="row_sparse"
            ),
        ).to("cuda")
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: spops_sdd_structure_aware(X, W_row_sparse, W_spops_row),
            quantiles=quantiles,
        )
    if provider == "spops_sputnik_block":
        W_spops_block = SparseWeights.from_dense(
            W,
            SparseMaskConfig(
                keep_ratio=sp, block_size=BLOCK_SIZE, sps_type="block_sparse"
            ),
        ).to("cuda")
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: spops_sdd_sputnik(X, W, W_spops_block), quantiles=quantiles
        )
    if provider == "triton_bs":
        op = prepare_triton_bs_op(X, W)
        X = X[None, None, ...]
        W = W[None, None, ...]
        ms, min_ms, max_ms = tn.testing.do_bench(
            lambda: triton_blocksparse_mm(X, W, op), quantiles=quantiles
        )

    gbps = lambda ms: 2 * s * h * o * 2 * 1e-9 / (ms * 1e-3)
    # return gbps(ms), gbps(max_ms), gbps(min_ms)
    return ms, max_ms, min_ms


benchmark.run(show_plots=True, print_data=True, save_path=".")
