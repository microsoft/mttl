import torch
from stk.matrix import Matrix

from mttl.models.modifiers.sparsity.spb_moe import functions


def sdd_adamerge(a, b, out_topo: Matrix, out_adaps: Matrix, layout):
    assert isinstance(a, torch.Tensor)
    assert isinstance(b, torch.Tensor)
    assert isinstance(out_topo, Matrix)
    assert out_topo.is_contiguous()
    assert isinstance(out_adaps, Matrix)
    assert out_adaps.data.is_contiguous()
    assert isinstance(layout, torch.Tensor)
    assert layout.is_contiguous()
    # essentially merged the adapters into a single Matrix()
    assert (
        out_adaps.shape[1] == out_topo.shape[1]
    ), "This performs sparse SDD of a and b, the output topo should have the same number of columns as the out_adaps"
    assert (
        out_adaps.shape[1] % b.size(1) == 0
    ), "The number of columns in out_adaps should be a multiple of the number of columns in b"

    out = functions.sdd_spsmerge(
        a,
        b,
        out_topo.size(),
        out_topo.data,
        out_topo.row_indices,
        out_topo.column_indices,
        out_topo.column_indices_t,
        out_topo.block_offsets_t,
        out_adaps.data,
        layout,
    )
    return Matrix(
        out_topo.size(),
        out,
        out_topo.row_indices,
        out_topo.column_indices,
        out_topo.offsets,
        out_topo.column_indices_t,
        out_topo.offsets_t,
        out_topo.block_offsets_t,
    )


def scattergather_adamerge(
    x,
    base_act,
    k,
    ada_weights,
    row_idxs,
    col_idxs,
    offsets,
    block_offsets_t,
    ada_block_size,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    padded_block_idxs,
    gates,
):

    out = functions.parallel_linear(
        x,
        base_act,
        k,
        ada_weights,
        row_idxs,
        col_idxs,
        offsets,
        block_offsets_t,
        ada_block_size,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        gates,
    )
    return out


def scattergather_adamerge_opt(
    x,
    base_act,
    k,
    ada_weights,
    row_idxs,
    col_idxs,
    offsets,
    block_offsets_t,
    ada_block_size,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    padded_block_idxs,
    gates,
):

    out = functions.parallel_linear_optimized(
        x,
        base_act,
        k,
        ada_weights,
        row_idxs,
        col_idxs,
        offsets,
        block_offsets_t,
        ada_block_size,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        padded_block_idxs,
        gates,
    )
    return out


BLOCK_M = 128  # expert token capacity


@torch.jit.script
def flatten_and_sort(expert_idxs: torch.Tensor):
    """
    Flattens a tensor of expert indices and sorts the flattened tensor.

    Args:
        expert_idxs (torch.Tensor): A tensor containing expert indices.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - sorted_expert_idxs: The flattened and sorted expert indices.
            - sorted_scattered_idxs: The indices that would sort the flattened tensor.
    """
    flattened_expert_idxs = expert_idxs.flatten()
    sorted_expert_idxs, sorted_scattered_idxs = torch.sort(flattened_expert_idxs)
    return sorted_expert_idxs, sorted_scattered_idxs


@torch.jit.script
def padded_block_indices(
    sorted_experts_idxs: torch.Tensor, k: int, N_BLOCK_SIZE: int = BLOCK_M
):
    """
    Compute padded block indices for sorted experts.

    This function calculates the indices of padded blocks for a given set of sorted expert indices.
    It ensures that the blocks are padded to a specified block size.

    Args:
        sorted_experts_idxs (torch.Tensor): A tensor containing the sorted indices of experts.
        k (int): The number of unique experts.
        N_BLOCK_SIZE (int, optional): The size of each block. Defaults to BLOCK_M.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - expanded_block_idxs (torch.Tensor): The indices of the expanded blocks.
            - expert_boundaries_end (torch.Tensor): The end boundaries of the experts.
    """
    expert_counts = torch.bincount(sorted_experts_idxs, minlength=k)
    padded_block_counts = ((expert_counts - 1) // N_BLOCK_SIZE) + 1
    padded_expert_block_end = padded_block_counts.cumsum(-1)
    expert_boundaries_end = expert_counts.cumsum(-1)
    expert_boundaries_start = expert_boundaries_end - expert_counts
    padded_expert_block_start = padded_expert_block_end - padded_block_counts
    block_idxs = torch.arange(
        padded_expert_block_end[-1],
        dtype=sorted_experts_idxs.dtype,
        device=sorted_experts_idxs.device,
    )
    block_mask = (block_idxs[:, None] < padded_expert_block_start) | (
        block_idxs[:, None] >= padded_expert_block_end
    )
    expanded_block_idxs = (
        N_BLOCK_SIZE * (block_idxs[:, None] - padded_expert_block_start)
        + expert_boundaries_start
    )
    expanded_block_idxs = expanded_block_idxs.masked_fill(block_mask, 0).sum(-1)
    return expanded_block_idxs, expert_boundaries_end
