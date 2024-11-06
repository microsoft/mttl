from typing import Any

import torch
from stk.backend.autocast import custom_bwd, custom_fwd
from stk.matrix import Matrix

from mttl.models.modifiers.sparsity.spb_moe.triton_kernels import (
    scatter2scatter_sparse,
    scatter2scatter_sparse_optimized,
)


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
        gates,
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


class ParalleLinear_optim(torch.autograd.Function):

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


parallel_linear_optimized = ParalleLinear_optim.apply
