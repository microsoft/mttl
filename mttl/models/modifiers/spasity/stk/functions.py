from typing import Any

import torch
from stk.backend.autocast import custom_bwd, custom_fwd
from stk.matrix import Matrix

import mttl.models.modifiers.spasity.stk.triton_kernels as backend
from mttl.models.modifiers.spasity.stk.scatter_moe_kernels import (
    scatter2scatter_sparse,
    scatter2scatter_sparse_optimized,
)


class RowIndices(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shape, data, offsets, column_indices):
        out = torch.empty(
            column_indices.shape,
            dtype=column_indices.dtype,
            device=column_indices.device,
        )
        backend.row_indices(shape, data, offsets, column_indices, out)
        return out


row_indices = RowIndices.apply


class SDD_SpMerge(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        lhs,
        rhs,
        shape,
        data,
        row_indices,
        column_indices,
        column_indices_t,
        block_offsets_t,
        adap_data,
        ada_maping,
    ):
        # note for later: here we will need ofdfsets transpose and offsets for the baclkward pass if we implement it
        out = torch.empty(data.shape, dtype=lhs.dtype, device=lhs.device)
        backend.sdd_spmerge(
            lhs, rhs, shape, out, row_indices, column_indices, adap_data, ada_maping
        )
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        raise NotImplementedError


sdd_spsmerge = SDD_SpMerge.apply


class PaddedGatherOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        ctx.save_for_backward(indices, bin_ids, bins, padded_bins)
        ctx.top_k = top_k
        return backend.padded_gather(
            x,
            indices,
            bin_ids,
            None,
            bins,
            padded_bins,
            top_k,
        )

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()

        indices, bin_ids, bins, padded_bins = ctx.saved_tensors
        out = backend.padded_scatter(
            grad,
            indices,
            bin_ids,
            None,
            bins,
            padded_bins,
            ctx.top_k,
        )
        return out, None, None, None, None, None


padded_gather = PaddedGatherOp.apply


class ParalleLinear(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
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
        gates
    ):

        output = scatter2scatter_sparse(
            X=x,
            base_act=base_act,
            ada_weights=ada_weights,
            row_idxs=row_idxs,
            col_idxs_t=col_idxs,
            offsets_t=offsets,
            block_offsets_t=block_offsets_t,
            ada_block=ada_block_size,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=k,
            gates=gates,
        )
        output = output.view(gates.size(0), gates.size(1), output.size(-1)).sum(
            1
        )  # this can be moved into kernel?
        return output


parallel_linear = ParalleLinear.apply



class ParalleLinear2(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
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
        gates=None,
    ):

        output = scatter2scatter_sparse_optimized(
            X=x,
            base_act=base_act,
            ada_weights=ada_weights,
            row_idxs=row_idxs,
            col_idxs_t=col_idxs,
            offsets_t=offsets,
            block_offsets_t=block_offsets_t,
            ada_block=ada_block_size,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=k,
            gates=gates,
        )
        output = output.view(gates.size(0), gates.size(1), output.size(-1)).sum(
            1
        )  # this can be moved into kernel?
        return output


parallel_linear_optimized = ParalleLinear2.apply