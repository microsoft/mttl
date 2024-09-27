from typing import List

import numpy as np
import torch

from mttl.models.modifiers.spasity import Matrix
from mttl.models.modifiers.spasity.stk import functions

# mostly taken/adapter from STK:  https://github.com/stanford-futuredata/stk


@torch.no_grad()
def row_indices(shape, data, offsets, column_indices):
    return functions.row_indices(shape, data, offsets, column_indices)


# TODO(tgale): Replace this helper with a custom kernel. This operation
# is much simpler to do than how it's currently implemented.
@torch.no_grad()
def _expand_for_blocking(idxs, blocking):
    # Duplicate for block column dimension.
    idxs = torch.reshape(idxs, [idxs.size()[0], 1, 2]).repeat(1, blocking, 1)

    # Update the column indices.
    idxs[:, :, 1] *= blocking
    idxs[:, :, 1] += torch.reshape(
        torch.arange(blocking, device=idxs.device), [1, blocking]
    )

    # Duplicate for block row dimension.
    idxs = torch.reshape(idxs, [idxs.size()[0], 1, blocking, 2])
    idxs = idxs.repeat(1, blocking, 1, 1)

    # Update the row indices.
    idxs[:, :, :, 0] *= blocking
    idxs[:, :, :, 0] += torch.reshape(
        torch.arange(blocking, device=idxs.device), [1, blocking, 1]
    )
    idxs = torch.reshape(idxs, [-1, 2])
    return idxs


# TODO(tgale): Add input type checking.
@torch.no_grad()
def to_dense(x):
    assert isinstance(x, Matrix)

    shape = (np.prod(x.shape[:-1]), x.shape[-1])
    row_idxs = x.row_indices.type(torch.int32)
    col_idxs = x.column_indices.type(torch.int32)
    indices = _expand_for_blocking(torch.stack([row_idxs, col_idxs], dim=1), x.blocking)
    indices = (indices[:, 0] * shape[1] + indices[:, 1]).type(torch.int64)

    out = torch.zeros(shape[0] * shape[1], dtype=x.dtype, device=x.device)
    out.scatter_(0, indices, x.data.flatten())
    return out.reshape(x.size())


@torch.no_grad()
def _mask(x, blocking=1):
    assert x.dim() == 2
    assert x.size()[0] % blocking == 0
    assert x.size()[1] % blocking == 0
    block_rows = x.size()[0] // blocking
    block_cols = x.size()[1] // blocking
    x = torch.reshape(x, [block_rows, blocking, block_cols, blocking])
    x = torch.sum(torch.abs(x), dim=(1, 3))
    return x != 0


# TODO(tgale): Add input type checking.
@torch.no_grad()
def to_sparse(x, blocking=1):
    m = _mask(x, blocking)

    # TODO(tgale): Set to appropriate type for input matrix.
    row_nnzs = torch.sum(m, dim=1).type(torch.int32)
    zeros = torch.zeros((1,), dtype=row_nnzs.dtype, device=row_nnzs.device)
    offsets = torch.cat([zeros, torch.cumsum(row_nnzs, dim=0)])
    offsets = offsets.type(torch.int32)

    indices = torch.nonzero(m).type(torch.int16)
    row_indices = indices[:, 0]
    column_indices = indices[:, 1]

    # Nonzero indices in the dense matrix.
    nonzero_indices = torch.nonzero(m)
    nonzero_indices = _expand_for_blocking(nonzero_indices, blocking)
    nonzero_indices = nonzero_indices[:, 0] * x.size()[1] + nonzero_indices[:, 1]

    # Gather the data and construct the sparse matrix.
    data = torch.gather(x.flatten(), dim=0, index=nonzero_indices)
    data = torch.reshape(data, [-1, blocking, blocking])
    return Matrix(x.size(), data, row_indices, column_indices, offsets)


@torch.no_grad()
def ones_like(x):
    return Matrix(
        x.size(), torch.ones_like(x.data), x.row_indices, x.column_indices, x.offsets
    )


def sum(x):
    assert isinstance(x, Matrix)
    return x.data.sum()


def merge_adapters(adapters: List[Matrix]) -> Matrix:
    """
    Merges a list of adapters into a single adapter along the second dimention.
    """
    col_indices_list = [adap.column_indices for adap in adapters]
    row_indices_list = [adap.row_indices for adap in adapters]
    offsets_list = [adap.offsets for adap in adapters]
    data_list = [adap.data for adap in adapters]

    num_rows = [offsets.numel() - 1 for offsets in offsets_list]
    assert all(
        num_rows[0] == num_rows[i] for i in range(1, len(num_rows))
    ), "All adapters must have the same number of rows"

    block_size = adapters[0].blocking
    K, N = adapters[0].size()
    col_offset = N // block_size  # assuming all have same number of cols
    n_adaps = len(adapters)

    adjusted_col_indices = []
    for e, col_idx in enumerate(col_indices_list):
        adjusted_col_indices.append(col_idx + e * col_offset)

    merged_col_indices = torch.cat(adjusted_col_indices)
    row_indices = torch.cat([adap.row_indices for adap in adapters], dim=0)
    data = torch.cat([adap.data for adap in adapters], dim=0)

    indices = torch.stack([row_indices, merged_col_indices], dim=1)

    if indices.is_cuda:
        indices = indices.cpu()

    # Convert to NumPy
    np_tensor = indices.numpy()
    # Perform lexsort: sort by second key first, then first key
    sorted_indices = np.lexsort((np_tensor[:, 1], np_tensor[:, 0]))

    data = data[sorted_indices].contiguous()
    row_indices = row_indices[sorted_indices].contiguous()
    col_indices = merged_col_indices[sorted_indices].contiguous()

    # recalculate offsets
    num_rows = max(num_rows)
    offsets = torch.zeros(num_rows + 1, dtype=torch.int32, device=row_indices.device)
    counts_per_row = torch.bincount(row_indices, minlength=num_rows)
    offsets[1:] = torch.cumsum(counts_per_row, dim=0)
    offsets = offsets.contiguous()

    return Matrix((K, n_adaps * N), data, row_indices, col_indices, offsets)


def create_ada_indices(
    row_indices, column_indices, ada_row_indices, ada_column_indices, device
):
    """ """
    nnz_blocks = len(row_indices)
    ada_block_map = {}
    for idx, (r, c) in enumerate(zip(ada_row_indices, ada_column_indices)):
        ada_block_map[(r.item(), c.item())] = idx

    ada_indices = torch.full((nnz_blocks,), -1, dtype=torch.int32, device=device)
    for pid in range(nnz_blocks):
        pid_m = row_indices[pid].item()
        pid_n = column_indices[pid].item()
        if (pid_m, pid_n) in ada_block_map:
            ada_indices[pid] = ada_block_map[(pid_m, pid_n)]
    return ada_indices


def create_ada_layout(matix: Matrix):
    """
    Creates a binary tensor that identifies if block exists in the adapter matrix
    """
    block_size = matix.blocking
    layout = (
        torch.ones(
            (matix.size()[0] // block_size, matix.size()[1] // block_size),
            dtype=torch.int32,
            device=matix.device,
        )
        * -1
    )
    blck = 0
    for r, c in zip(matix.row_indices, matix.column_indices):
        layout[r.item(), c.item()] = blck
        blck += 1
    return layout.contiguous()
