import pytest
import stk
import stk.ops
import torch
from stk.matrix import Matrix

from mttl.models.modifiers.sparsity.sparse_utils import stk_matrix_utils as matrix_ops


@pytest.mark.parametrize(
    "K, rows, cols, sparsity, blocking",
    [
        (2, 8, 16, 0.5, 1),
        (2, 8, 16, 0.5, 4),
    ],
)
def test_layout_creation(self, K, rows, cols, sparsity, blocking):
    adaps = [
        matrix_ops._dense_and_sparse(rows, cols, sparsity, blocking, torch.float16)
        for _ in range(K)
    ]
    adaps_sparse = [adap[1] for adap in adaps]
    # adaps_dense = [adap[0] for adap in adaps]

    merged_adaps_matrix: Matrix = matrix_ops.merge_adapters(adaps_sparse)
    layout = matrix_ops.create_ada_layout(merged_adaps_matrix)
    assert layout.max() == merged_adaps_matrix.data.size(0) - 1


if __name__ == "__main__":
    pytest.main([__file__])
