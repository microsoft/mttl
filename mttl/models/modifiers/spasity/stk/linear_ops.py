import torch
from stk.matrix import Matrix

from mttl.models.modifiers.spasity.stk import functions


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
