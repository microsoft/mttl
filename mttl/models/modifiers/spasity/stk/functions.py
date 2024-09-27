import torch
from stk.backend.autocast import custom_bwd, custom_fwd
from stk.matrix import Matrix

import mttl.models.modifiers.spasity.stk.triton_kernels as backend


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
