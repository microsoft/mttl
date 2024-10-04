import torch
import triton
import triton.language as tl
from torch.nn import functional as F

BLOCK_M = 128


def _scatter2scatter_configs():
    return [
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_stages=1, num_warps=1),
    ]


@triton.autotune(
    configs=_scatter2scatter_configs(),
    key=["M", "N", "K"],
)
@triton.heuristics(
    {
        "NO_K_MASK": lambda args: (args["K"] % args["BLOCK_K"]) == 0,
        "NO_N_MASK": lambda args: (args["N"] % args["BLOCK_N"]) == 0,
        "ADA_BLCKS_PER_TILE_K": lambda args: args["BLOCK_K"] // args["ADA_BLOCK"],
        "ADA_BLCKS_PER_TILE_N": lambda args: args["BLOCK_N"] // args["ADA_BLOCK"],
    }
)
@triton.jit
def _scatter2scatter(
    X_ptr,
    stride_xm,
    stride_xk,
    W_ptr,
    stride_wk,
    stride_wn,
    adaW,  # n_exp x ada_block x ada_block
    ada_layout,
    stride_layout_e,
    stride_layout_m,
    stride_layout_n,
    Y_ptr,
    stride_ym,
    stride_yn,
    sorted_scattered_idxs,
    sorted_expert_idxs,
    padded_block_idxs,
    FAN_OUT,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr,
    x_grouped: tl.constexpr,
    y_grouped: tl.constexpr,
    NO_K_MASK: tl.constexpr,
    NO_N_MASK: tl.constexpr,
    ADA_BLOCK: tl.constexpr,
    ADA_BLCKS_PER_TILE_K: tl.constexpr,  # how many ada blocks in one tile in K direction
    ADA_BLCKS_PER_TILE_N: tl.constexpr,  # how many ada blocks in one tile in N direction
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(
        N, BLOCK_N
    )  # is 2? numbe of blocks per expert's output dimension
    M_block_id = (
        pid // N_BLOCK_COUNT
    )  # which expert are we in? (actually block, since there might be multiple blocks per expert)
    N_block_id = pid % N_BLOCK_COUNT  # which block in the out. dim are we in?
    # Determine the block indices along the M and N dimensions for this program.

    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(
        padded_block_idxs + M_block_id
    )  # Load the index of the  starting token for this block
    # M_block = tl.max_contiguous((block_start_idx + M_range) % OUT_M, BLOCK_M)
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)  # max tokens
    E_idxs = tl.load(
        sorted_expert_idxs + M_block, mask=M_block < (FAN_OUT * M), other=E
    )  # expert_idxs_ptr is sorted by expert! so this loads expert indices of tokens
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    M_idx = tl.load(sorted_scattered_idxs + M_block, mask=E_mask, other=0)
    if x_grouped:
        M_in_idx = M_block
    else:
        M_in_idx = M_idx // FAN_OUT

    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx

    K_block = tl.arange(0, BLOCK_K)

    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N

    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :] * stride_wn

    L_BLOCK_K = tl.arange(0, ADA_BLCKS_PER_TILE_K)
    L_BLOCK_N = tl.arange(0, ADA_BLCKS_PER_TILE_N)
    additive_idx_blocks = (tl.arange(0, ADA_BLOCK))[:, None] * ADA_BLOCK + (
        tl.arange(0, ADA_BLOCK)
    )[None, :]
    L_blck_ptrs = (
        ada_layout
        + L_BLOCK_K[:, None] * stride_layout_m
        + L_BLOCK_N[None, :] * stride_layout_n
        + N_block_id * ADA_BLCKS_PER_TILE_N
        + E_idx * stride_layout_e
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(0, iters):
        if NO_K_MASK:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            if NO_N_MASK or K_block_id < (iters - 1):
                w = tl.load(W_blk_ptrs)
            else:
                w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = (K_block_id * BLOCK_K + K_block) < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])

        layout_tile = tl.load(L_blck_ptrs)  # 2 x 8
        # BETTER TO RESAHPE MEMORY ADDRESSES, NOT THE LOADED DATA?
        mask = layout_tile >= 0
        base_addresses = adaW + (layout_tile * (ADA_BLOCK * ADA_BLOCK))
        full_addresses = (
            base_addresses[:, None, :, None] + additive_idx_blocks[None, :, None, :]
        )
        full_addresses = full_addresses.reshape(
            ADA_BLCKS_PER_TILE_K * ADA_BLOCK, ADA_BLCKS_PER_TILE_N * ADA_BLOCK
        )
        mask = mask[:, None, :, None] * (
            tl.zeros((1, ADA_BLOCK, 1, ADA_BLOCK), dtype=ACC_TYPE) + 1.0
        )
        mask = (
            mask.reshape(
                ADA_BLCKS_PER_TILE_K * ADA_BLOCK, ADA_BLCKS_PER_TILE_N * ADA_BLOCK
            )
            > 0.0
        )

        adaW_tile = tl.load(
            full_addresses,
            mask=mask,
            other=0.0,
        )
        w = (
            w + adaW_tile
        )  # .reshape(ADA_BLCKS_PER_TILE_K * ADA_BLOCK, ADA_BLCKS_PER_TILE_N * ADA_BLOCK)
        L_blck_ptrs += ADA_BLCKS_PER_TILE_K * stride_layout_m
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc += tl.dot(x, w, out_dtype=ACC_TYPE)

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])


def scatter2scatter(
    X,
    W,
    ada_weights,
    ada_block,
    ada_layout,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    k,
    padded_block_idxs,
    x_grouped=False,
    y_grouped=False,
    out=None,
):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k
    # Pre-kernel setup
    x_dim = X.size(-1)
    y_dim = W.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        O = torch.empty((L_scattered, y_dim), device=X.device, dtype=X.dtype)
        # O = torch.empty_like(ada_weights, device=X.device, dtype=ada_weights.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        O = out

    def grid(META):
        grid_num = (
            padded_block_idxs.size(0) * triton.cdiv(META["N"], META["BLOCK_N"]),
        )
        return grid_num

    assert _scatter2scatter_configs()[0].kwargs["BLOCK_N"] % ada_block == 0
    assert _scatter2scatter_configs()[0].kwargs["BLOCK_K"] % ada_block == 0
    assert (ada_layout.size(1) * ada_block) % W.size(-1) == 0

    M, K = X.size()
    N = y_dim
    E = (ada_layout.size(1) * ada_block) // W.size(-1)
    ada_layout_stride_e = N // ada_block
    # sorted_expert_idxs = sorted_expert_idxs.to(torch.int32)
    # sorted_scattered_idxs = sorted_scattered_idxs.to(torch.int32)
    # padded_block_idxs = padded_block_idxs.to(torch.int32)

    # with torch.cuda.device(X.device):
    _scatter2scatter[grid](
        X,
        X.stride(0),
        X.stride(1),
        W,
        W.stride(0),
        W.stride(1),
        ada_weights,  # n_exp x ada_block x ada_block
        ada_layout,
        ada_layout_stride_e,
        ada_layout.stride(0),
        ada_layout.stride(1),
        O,
        O.stride(0),
        O.stride(1),
        sorted_scattered_idxs,
        sorted_expert_idxs,
        padded_block_idxs,
        FAN_OUT=k,
        M=M,
        K=K,
        N=N,
        E=E,
        BLOCK_M=BLOCK_M,
        ACC_TYPE=tl.float32,
        allow_tf32=True,
        x_grouped=x_grouped,
        y_grouped=y_grouped,
        ADA_BLOCK=ada_block,
    )
    return O


def _scatter2scatter_sp_configs():
    return [
        # triton.Config({"BLOCK_K": 128}, num_stages=4, num_warps=4),
    ]


@triton.autotune(
    configs=_scatter2scatter_sp_configs(),
    key=["M", "N"],
)
@triton.jit
def _scatter2scatter_sp(
    X_ptr,
    stride_xm,
    stride_xk,
    gates,
    adaW,  # n_exp x ada_block x ada_block
    adaW_stride_e,
    adaW_stride_m,
    adaW_stride_n,
    base_acts,
    column_indices_t,  # gives the row index column by column (row indexes sorted by column starting witht he first one etc.)
    column_indices_t_offset,
    offsets_t,  # offsets for columns: i.e. the diff between two consecutive gives the number of blocks per column. It indexes column_indices_t, block_offsets_t
    offsets_t_offset,
    block_offsets_t,  # indices of blocks sorted by column
    block_offsets_t_offset,
    Y_ptr,
    stride_ym,
    stride_yn,
    # OW,
    sorted_scattered_idxs,
    sorted_expert_idxs,
    padded_block_idxs,
    FAN_OUT,
    M,
    N,
    E,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    N_BLOCK_COUNT = tl.cdiv(
        N, BLOCK_N
    )  # is 2? numbe of blocks per expert's output dimension
    M_block_id = (
        pid // N_BLOCK_COUNT
    )  # which expert are we in? (actually block, since there might be multiple blocks per expert)
    N_block_id = pid % N_BLOCK_COUNT  # which block in the out. dim are we in?
    # Determine the block indices along the M and N dimensions for this program.

    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(
        padded_block_idxs + M_block_id
    )  # Load the index of the  starting token for this block
    # M_block = tl.max_contiguous((block_start_idx + M_range) % OUT_M, BLOCK_M)
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)  # max tokens
    E_idxs = tl.load(
        sorted_expert_idxs + M_block, mask=M_block < (FAN_OUT * M), other=E
    )  # expert_idxs_ptr is sorted by expert! so this loads expert indices of tokens
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    M_idx = tl.load(sorted_scattered_idxs + M_block, mask=E_mask, other=0)
    M_in_idx = M_idx // FAN_OUT
    M_out_idx = M_idx

    K_block = tl.arange(0, BLOCK_K)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    start_inx = tl.load(offsets_t + (E_idx * offsets_t_offset) + N_block_id)
    end_inx = tl.load(offsets_t + (E_idx * offsets_t_offset) + N_block_id + 1)
    num_blocks_column = end_inx - start_inx
    iters = num_blocks_column  # tl.cdiv(num_blocks_column, tl.cdiv(BLOCK_K, ADA_BLOCK)) # n_blocks_column
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    gate = tl.load(gates + M_idx, mask=E_mask)

    if iters > 0:
        # pointers to dense matrix
        X_blk_ptr = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk

        # pointers to sparse matrix
        rn = tl.arange(0, BLOCK_N)  # ...16
        rbk = tl.arange(0, BLOCK_K)  # ... 16
        W_blk_ptr = (
            adaW
            + (rbk[:, None] * adaW_stride_m)
            + (rn[None, :] * adaW_stride_n)
            + (E_idx * adaW_stride_e)
        )
        BLOCK_SIZE = BLOCK_K * BLOCK_N
        ak_block_incr = stride_xk * BLOCK_K

        # OW_block_ptr = OW + (rbk[:, None] * adaW_stride_m) + (rn[None, :] * adaW_stride_n) + (E_idx * adaW_stride_e)

        for K_block_id in range(0, iters):
            X = (
                X_blk_ptr
                + tl.load(
                    column_indices_t
                    + (E_idx * column_indices_t_offset)
                    + start_inx
                    + K_block_id
                )
                * ak_block_incr
            )

            W = (
                W_blk_ptr
                + tl.load(
                    block_offsets_t
                    + (E_idx * block_offsets_t_offset)
                    + start_inx
                    + K_block_id
                )
                * BLOCK_SIZE
            )
            # OWW = OW_block_ptr + tl.load(block_offsets_t + (E_idx * block_offsets_t_offset) + start_inx + K_block_id) * BLOCK_SIZE

            x = tl.load(X, mask=E_mask[:, None])
            w = tl.load(W, mask=N_mask[None, :])
            acc += tl.dot(x, w, out_dtype=ACC_TYPE)

            # tl.store(OWW, w)

    base_act_ptr = (
        base_acts + M_in_idx[:, None] * stride_ym + N_block[None, :] * stride_yn
    )
    base_act = tl.load(base_act_ptr, mask=E_mask[:, None] & N_mask[None, :])
    acc *= gate[:, None]
    acc += base_act

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])


def scatter2scatter_sparse(
    X,
    base_act,
    ada_weights,
    row_idxs,
    col_idxs_t,
    ada_block,
    offsets_t,
    block_offsets_t,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    k,
    padded_block_idxs,
    gates,
    out=None,
):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k
    assert X.is_contiguous()
    assert base_act.is_contiguous()
    assert ada_weights.is_contiguous()
    assert row_idxs.is_contiguous()
    assert col_idxs_t.is_contiguous()
    assert offsets_t.is_contiguous()
    assert block_offsets_t.is_contiguous()
    assert sorted_expert_idxs.is_contiguous()
    assert sorted_scattered_idxs.is_contiguous()
    assert padded_block_idxs.is_contiguous()
    assert gates.is_contiguous()

    # Pre-kernel setup
    x_dim = X.size(-1)
    y_dim = base_act.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        O = torch.empty((L_scattered, y_dim), device=X.device, dtype=X.dtype)
        # O = torch.empty_like(ada_weights, device=X.device, dtype=ada_weights.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        O = out

    # OW = torch.empty_like(ada_weights, device=X.device, dtype=ada_weights.dtype)
    def grid(META):
        grid_num = (
            padded_block_idxs.size(0) * triton.cdiv(META["N"], META["BLOCK_N"]),
        )
        return grid_num

    M, K = X.size()
    N = y_dim
    E = ada_weights.size(0)
    with torch.cuda.device(X.device):
        _scatter2scatter_sp[grid](
            X,
            X.stride(0),
            X.stride(1),
            gates,
            ada_weights,  # n_exp x ada_block x ada_block
            ada_weights.stride(0),
            ada_weights.stride(2),
            ada_weights.stride(3),
            base_act,
            col_idxs_t,
            col_idxs_t.stride(0),
            offsets_t,  # column offsets shapre is (E, N//ada_block + 1)
            offsets_t.stride(0),
            block_offsets_t,
            block_offsets_t.stride(0),
            O,
            O.stride(0),
            O.stride(1),
            # OW,
            sorted_scattered_idxs,
            sorted_expert_idxs,
            padded_block_idxs,
            FAN_OUT=k,
            M=M,
            N=N,
            E=E,
            BLOCK_M=BLOCK_M,
            BLOCK_K=ada_block,
            BLOCK_N=ada_block,
            ACC_TYPE=tl.float32,
        )
        return O


@triton.autotune(
    configs=[
        triton.Config({"GROUP_M": 1, "BLOCK_M": 128}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 4, "BLOCK_M": 128}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 32, "BLOCK_M": 128}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 128, "BLOCK_M": 128}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 1, "BLOCK_M": 64}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 4, "BLOCK_M": 64}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 32, "BLOCK_M": 64}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 128, "BLOCK_M": 64}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 1, "BLOCK_M": 256}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 4, "BLOCK_M": 256}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 32, "BLOCK_M": 256}, num_stages=4, num_warps=4),
        triton.Config({"GROUP_M": 128, "BLOCK_M": 256}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "E"],
)
@triton.jit
def _scatter2scatter_sp_optimized(
    X_ptr,
    stride_xm,
    stride_xk,
    gates,
    adaW,  # n_exp x ada_block x ada_block
    adaW_stride_e,
    adaW_stride_m,
    adaW_stride_n,
    base_acts,
    column_indices_t,  # gives the row index column by column (row indexes sorted by column starting witht he first one etc.)
    column_indices_t_offset,
    offsets_t,  # offsets for columns: i.e. the diff between two consecutive gives the number of blocks per column. It indexes column_indices_t, block_offsets_t
    offsets_t_offset,
    block_offsets_t,  # indices of blocks sorted by column
    block_offsets_t_offset,
    Y_ptr,
    stride_ym,
    stride_yn,
    sorted_scattered_idxs,
    sorted_expert_idxs,
    padded_block_idxs,
    FAN_OUT: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    num_pid_m = tl.num_programs(0)
    num_pid_n = tl.num_programs(1)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_M)

    M_block_id = pid_m  # which expert are we in? (actually block, since there might be multiple blocks per expert)
    N_block_id = pid_n  # which block in the out. dim are we in?
    M_range = tl.arange(0, BLOCK_M)
    block_start_idx = tl.load(
        padded_block_idxs + M_block_id
    )  # Load the index of the  starting token for this block
    # M_block = tl.max_contiguous((block_start_idx + M_range) % OUT_M, BLOCK_M)
    M_block = tl.max_contiguous(block_start_idx + M_range, BLOCK_M)  # max tokens
    E_idxs = tl.load(
        sorted_expert_idxs + M_block, mask=M_block < (FAN_OUT * M), other=E
    )  # expert_idxs_ptr is sorted by expert! so this loads expert indices of tokens
    E_idx = tl.min(E_idxs)
    E_mask = E_idxs == E_idx
    M_idx = tl.load(sorted_scattered_idxs + M_block, mask=E_mask, other=0)
    M_in_idx = M_idx // FAN_OUT
    M_out_idx = M_idx

    K_block = tl.arange(0, BLOCK_K)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    start_inx = tl.load(offsets_t + (E_idx * offsets_t_offset) + N_block_id)
    end_inx = tl.load(offsets_t + (E_idx * offsets_t_offset) + N_block_id + 1)
    num_blocks_column = end_inx - start_inx
    iters = num_blocks_column  # tl.cdiv(num_blocks_column, tl.cdiv(BLOCK_K, ADA_BLOCK)) # n_blocks_column
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    gate = tl.load(gates + M_idx, mask=E_mask)

    if iters > 0:
        # pointers to dense matrix
        X_blk_ptr = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :] * stride_xk
        # pointers to sparse matrix
        rn = tl.arange(0, BLOCK_N)  # ...16
        rbk = tl.arange(0, BLOCK_K)  # ... 16
        W_blk_ptr = (
            adaW
            + (rbk[:, None] * adaW_stride_m)
            + (rn[None, :] * adaW_stride_n)
            + (E_idx * adaW_stride_e)
        )
        BLOCK_SIZE = BLOCK_K * BLOCK_N
        ak_block_incr = stride_xk * BLOCK_K

        for K_block_id in range(0, iters):
            X = (
                X_blk_ptr
                + tl.load(
                    column_indices_t
                    + (E_idx * column_indices_t_offset)
                    + start_inx
                    + K_block_id
                )
                * ak_block_incr
            )

            W = (
                W_blk_ptr
                + tl.load(
                    block_offsets_t
                    + (E_idx * block_offsets_t_offset)
                    + start_inx
                    + K_block_id
                )
                * BLOCK_SIZE
            )

            x = tl.load(X, mask=E_mask[:, None])
            w = tl.load(W, mask=N_mask[None, :])
            acc += tl.dot(x, w, out_dtype=ACC_TYPE)

    base_act_ptr = (
        base_acts + M_in_idx[:, None] * stride_ym + N_block[None, :] * stride_yn
    )
    base_act = tl.load(base_act_ptr, mask=E_mask[:, None] & N_mask[None, :])
    acc *= gate[:, None]
    acc += base_act

    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] * stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :])
    # tl.atomic_add(Y_blk_ptrs, acc, mask=E_mask[:, None] & N_mask[None, :], scope="cta")


def scatter2scatter_sparse_optimized(
    X,
    base_act,
    ada_weights,
    row_idxs,
    col_idxs_t,
    ada_block,
    offsets_t,
    block_offsets_t,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    k,
    padded_block_idxs,
    gates,
    out=None,
):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k
    assert X.is_contiguous()
    assert base_act.is_contiguous()
    assert ada_weights.is_contiguous()
    assert row_idxs.is_contiguous()
    assert col_idxs_t.is_contiguous()
    assert offsets_t.is_contiguous()
    assert block_offsets_t.is_contiguous()
    assert sorted_expert_idxs.is_contiguous()
    assert sorted_scattered_idxs.is_contiguous()
    assert padded_block_idxs.is_contiguous()
    assert gates.is_contiguous()

    # Pre-kernel setup
    x_dim = X.size(-1)
    y_dim = base_act.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        O = torch.zeros((L_scattered, y_dim), device=X.device, dtype=X.dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        O = out

    def grid(META):
        grid_num = (
            padded_block_idxs.size(0),
            triton.cdiv(META["N"], META["BLOCK_N"]),
        )
        return grid_num

    M, K = X.size()
    N = y_dim
    E = ada_weights.size(0)
    with torch.cuda.device(X.device):
        _scatter2scatter_sp_optimized[grid](
            X,
            X.stride(0),
            X.stride(1),
            gates,
            ada_weights,  # n_exp x ada_block x ada_block
            ada_weights.stride(0),
            ada_weights.stride(2),
            ada_weights.stride(3),
            base_act,
            col_idxs_t,
            col_idxs_t.stride(0),
            offsets_t,  # column offsets shapre is (E, N//ada_block + 1)
            offsets_t.stride(0),
            block_offsets_t,
            block_offsets_t.stride(0),
            O,
            O.stride(0),
            O.stride(1),
            # OW,
            sorted_scattered_idxs,
            sorted_expert_idxs,
            padded_block_idxs,
            FAN_OUT=k,
            M=M,
            N=N,
            E=E,
            BLOCK_K=ada_block,
            BLOCK_N=ada_block,
            ACC_TYPE=tl.float32,
        )
        return O
