import unittest

import stk
import stk.ops
import torch
from absl.testing import parameterized
from stk.matrix import Matrix

from mttl.models.modifiers.spasity.stk import matrix_ops


def _dense_and_sparse(rows, cols, sparsity, blocking, dtype, std=0.1):
    mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * std * mask).type(dtype)
    sparse = stk.ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return (
        dense.to(cuda_device).requires_grad_(True),
        sparse.to(cuda_device).requires_grad_(True),
    )

@parameterized.parameters(
    (2, 8, 16, 0.5, 1),
    (2, 8, 16, 0.5, 4)
    )
class MatrixOpsTest(parameterized.TestCase):        
    def test_layout_creation(self, K, rows, cols, sparsity, blocking):
        adaps = [_dense_and_sparse(rows, cols, sparsity, blocking, torch.float16) for _ in range(K)]
        adaps_sparse = [adap[1] for adap in adaps]
        # adaps_dense = [adap[0] for adap in adaps]
        
        merged_adaps_matrix: Matrix = matrix_ops.merge_adapters(adaps_sparse)
        layout = matrix_ops.create_ada_layout(merged_adaps_matrix)
        assert layout.max() == merged_adaps_matrix.data.size(0) - 1
            


if __name__ == '__main__':
    unittest.main()
