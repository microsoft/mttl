from typing import List

import numpy as np
import stk.ops
import torch
from stk.matrix import Matrix


def _dense(rows, cols, dtype, std=0.1):
    cuda_device = torch.device("cuda")
    out = (torch.randn(rows, cols) * std).type(dtype)
    return out.to(cuda_device).requires_grad_(True)


def _dense_and_sparse(rows, cols, sparsity, blocking, dtype, std=0.1):
    mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * std * mask).type(dtype)
    sparse = stk.ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return (
        dense.to(cuda_device).requires_grad_(True),
        sparse.to(cuda_device).requires_grad_(True),
    )


def _merge_adapters(adapters: List[Matrix]) -> Matrix:
    """
    Merges a list of adapters into a single adapter along the second dimention.
    Also changes the block size by padding blocks iwht 0s if necessary.

    """
    col_indices_list = [adap.column_indices.to(torch.int32) for adap in adapters]
    # row_indices_list = [adap.row_indices for adap in adapters]
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
    row_indices = torch.cat(
        [adap.row_indices.to(torch.int32) for adap in adapters], dim=0
    )
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


def change_block_size(M: Matrix, new_blk_size) -> Matrix:
    raise NotImplementedError("change_block_size is not implemented yet")
    return


def merge_adapters(adapters: List[Matrix], blk_size=None) -> Matrix:
    """
    Merges a list of adapters into a single adapter along the second dimention.
    Also changes the block size by padding blocks iwht 0s if necessary.

    """

    out = _merge_adapters(
        adapters
    )  # merges the adapters into a single Matrix() without changing the block size
    if blk_size is not None:
        out = change_block_size(out, blk_size)
    return out


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
