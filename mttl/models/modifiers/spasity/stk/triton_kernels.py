from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class TritonConfig:
    BLOCK_M: int = 16  # 128
    BLOCK_N: int = 16  # 128
    BLOCK_K: int = 16  # 32
    # BLOCK_SIZE: int = 128  # block size in the output matrix?
    NUM_STAGES: int = 4
    NUM_WARPS: int = 4


def _validate_matmul_dims(M: int, K: int, N: int):
    error_string = "incompatible dimensions: tensor has dim with length: {}, which must be divisible by {}"
    assert M % TritonConfig.BLOCK_M == 0, error_string.format(M, TritonConfig.BLOCK_M)
    assert K % TritonConfig.BLOCK_K == 0, error_string.format(K, TritonConfig.BLOCK_K)
    assert N % TritonConfig.BLOCK_N == 0, error_string.format(N, TritonConfig.BLOCK_N)


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config(
            {
                "BLOCK_M": TritonConfig.BLOCK_M,
                "BLOCK_N": TritonConfig.BLOCK_N,
                "BLOCK_K": TritonConfig.BLOCK_K,
                # "BLOCK_SIZE": TritonConfig.BLOCK_SIZE,
            },
            num_stages=TritonConfig.NUM_STAGES,
            num_warps=TritonConfig.NUM_WARPS,
        ),
    ],
    key=["M", "N", "K"], # uses these keys to decide wether to re-evaluate the choise of best config
)
@triton.jit  # this is understood
def _sdd_adamerge(
    A,
    B,
    S,
    OUT,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    row_indices,
    column_indices,
    layout,
    stride_layout_m,
    stride_layout_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)  # in triton only control thread blocks
    pid_m = tl.load(
        row_indices + pid
    )  # row index of the block in the output matrix that is being computed by this thread block
    pid_n = tl.load(
        column_indices + pid
    )  # column index of the block in the output matrix that is being computed by this thread block
    rm = pid_m * BLOCK_M + tl.arange(
        0, BLOCK_M
    )  # the actual row indices in the output matrix
    rn = pid_n * BLOCK_N + tl.arange(
        0, BLOCK_N
    )  # the actual column indices in the output matrix
    ram = tl.max_contiguous(
        tl.multiple_of(rm % M, BLOCK_M), BLOCK_M
    )  # optimizes memory throughput by ensuring that the memory accesses are contiguous
    rbn = tl.max_contiguous(
        tl.multiple_of(rn % N, BLOCK_N), BLOCK_N
    )  # optimizes memory throughput by ensuring that the memory accesses are contiguous
    rk = tl.arange(0, BLOCK_K)  # innialize inner dimention range for the current block
    BLOCK_ELEMENTS = BLOCK_M * BLOCK_N  # BLOCK_SIZE * BLOCK_SIZE
    cm = tl.arange(0, BLOCK_M)
    cn = tl.arange(0, BLOCK_N)
    # pointers
    A = A + (
        ram[:, None] * stride_am + rk[None, :] * stride_ak
    )  # BLOCK_M x BLOCK_K pointes to the dense matrix A for loading
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    # do matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        s_blck = tl.load(layout + k * stride_layout_m + pid_n * stride_layout_n)
        mask = s_blck >= 0
        s_blck = tl.where(mask, s_blck, 0)
        s_ptr = (
            S
            + s_blck * BLOCK_ELEMENTS
            + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
        )
        s = tl.load(s_ptr)
        s = tl.where(mask[None, None], s, tl.zeros_like(s))
        b = b + s
        acc += tl.dot(a, b)  # this should be using tensor cores on A100
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    # Store to sparse matrix
    acc = acc.to(OUT.dtype.element_ty)
    # remember, in OUT we only store the non-zero elements, so no need to map it to dense matrix
    OUT = (
        OUT + pid * BLOCK_ELEMENTS + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    )
    tl.store(OUT, acc, mask=True)


@triton.jit
def _row_indices_kernel(offsets, out):
    pid = tl.program_id(0)
    row_offset = tl.load(offsets + pid)
    nnz_blocks = tl.load(offsets + pid + 1) - row_offset
    for nnz_block in range(nnz_blocks):
        tl.store(out + row_offset + nnz_block, pid)


def row_indices(shape, data, offsets, column_indices, out):
    block_rows = len(offsets) - 1
    _row_indices_kernel[(block_rows,)](offsets, out)


def sdd_spmerge(
    x,
    base_weights,
    shape,
    out,
    row_indices,
    column_indices,
    ada_data,
    ada_layout,  #
):
    # E is the number of experts
    # ada_data is (E x n_blocks_per_e) x block_size x block_size
    # base_weights is dense matrix of shape (K, (expert_out_dim  x E)
    # ada_row_indices is (E x n_blocks_per_e)
    # ada_column_indices is (E x n_blocks_per_e)
    # base_weights.shape[1  = expert out dim.

    assert x.shape[1] == base_weights.shape[0], "incompatible dimensions"
    M, K = x.shape
    _, N = base_weights.shape
    assert (
        shape[1] & N == 0
    ), "RHS out dimension must be divisible by base weights output dim."
    E = shape[1] // N
    block_size = ada_data.shape[1]

    _validate_matmul_dims(M, K, N)

    if out.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        ACC_TYPE = tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {out.dtype}")
    
    # launch kernel
    nnz_blocks = len(row_indices)
    grid = lambda META: (nnz_blocks,)  # this just alunches 61 threadblocks

    stride_am, stride_ak = x.stride(0), x.stride(1)
    stride_bk, stride_bn = base_weights.stride(0), base_weights.stride(1)

    _sdd_adamerge[grid](
        x,
        base_weights,
        ada_data,
        out,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        out.stride(1),
        out.stride(2),
        row_indices,
        column_indices,
        ada_layout,
        ada_layout.stride(0),
        ada_layout.stride(1),
        ACC_TYPE=ACC_TYPE,
    )
