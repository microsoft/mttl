from typing import List

import numpy as np
import torch
from stk.matrix import Matrix


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
