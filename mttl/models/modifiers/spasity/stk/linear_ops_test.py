import itertools
import os
import unittest

import numpy as np
import stk
import torch
from absl.testing import parameterized
from stk.matrix import Matrix

from mttl.models.modifiers.spasity.stk import linear_ops, matrix_ops

# os.environ["TRITON_INTERPRET"] = "1"


def allclose(x, y, pct=0.25):
    mask = torch.isclose(x, y, rtol=5e-2)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


blocksize = 16
# An assortment of problems designed to make sure
# the bindings are operating correctly.
_MATRIX_SIZES = (
    (128, 128, 128, 0.8),
    (128, 128, 64, 0.8),
    (128, 128, 128, 0.0),
    # (256, 256, 256, 0.5),
    (2048, 1024, 512, 0.8),
    # (512, 128, 128, 0.0),
    # (128, 128, 512, 0.0),
    # (1024, 512, 512, 0.0),
    # (1024, 512, 512, 0.5),
    # (1024, 512, 512, 0.75),
    # (512, 512, 1024, 0.0),
    # (512, 512, 1024, 0.5),
    # (512, 512, 1024, 0.75),
    # (1024, 1024, 1024, 0.0),
    # (1024, 1024, 1024, 0.5),
    (1024, 1024, 1024, 0.75),
)

_TRANSPOSE = (
    (False, False),
    # (False, True),
    # (True, False),
    # (True, True),
)

_DTYPE = (torch.float16, torch.bfloat16)


def _generate_testcases():
    testcases = itertools.product(_MATRIX_SIZES, _TRANSPOSE, _DTYPE)
    testcases = [
        (*size, *trans, blocksize, dtype) for (size, trans, dtype) in testcases
    ]
    return testcases


_LINEAR_OP_TESTS = _generate_testcases()


def _dense_and_sparse(rows, cols, sparsity, blocking, dtype, std=0.1):
    mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
    dense = (torch.randn(rows, cols) * std * mask).type(dtype)
    sparse = stk.ops.to_sparse(dense, blocking)
    cuda_device = torch.device("cuda")
    return (
        dense.to(cuda_device).requires_grad_(True),
        sparse.to(cuda_device).requires_grad_(True),
    )


def _dense(rows, cols, dtype, std=0.1):
    cuda_device = torch.device("cuda")
    out = (torch.randn(rows, cols) * std).type(dtype)
    return out.to(cuda_device).requires_grad_(True)


def _dense_2x(rows, cols, dtype):
    a = _dense(rows, cols, dtype)
    return a, a.detach().requires_grad_(True)


def _mmm_with_adapters(a, W_base, topo, adapters):
    b = W_base.repeat(1, len(adapters))
    adaps_as_dense = [stk.ops.to_dense(adap) for adap in adapters]
    b = b + torch.cat(adaps_as_dense, dim=1)
    mask = stk.ops.to_dense(stk.ops.ones_like(topo))
    return torch.mm(a, b) * mask


@parameterized.parameters(*_LINEAR_OP_TESTS)
class LinearOpsTest(parameterized.TestCase):
    def testLinearOps_Sdd_wAdapters(
        self, m, k, n, sparsity, trans_a, trans_b, blocking, dtype
    ):
        if trans_a or trans_b:
            return
        # Construct the operands.
        # This tests the use-case where we have base weights and a bunch of adapters. We perform SDD of input x with base weights, but block-ssparse adapters are merged into the base weights first.

        a_shape = (m, k)
        a, acp = _dense_2x(*a_shape, dtype)

        n_adaps = 10
        adapters = [
            _dense_and_sparse(*(k, n), sparsity, blocking, dtype)[1]
            for _ in range(n_adaps)
        ]
        # merge all adapters into a single sparse Matrix()
        adaps: Matrix = matrix_ops.merge_adapters(adapters)

        out_shape = (m, n * n_adaps)
        _, out_topo = _dense_and_sparse(*out_shape, sparsity, blocking, dtype)
        # create a mapping from out_topo to adaps, indicating whether each out_topo bvlock needs to be merged with an adapter block, and if so which one
        layout = matrix_ops.create_ada_layout(adaps)

        w_shape = (k, n)
        W_base, W_basecp = _dense_2x(*w_shape, dtype)
        # Execute the matmul.
        out = linear_ops.sdd_adamerge(a, W_base, out_topo, adaps, layout)
        expected_out = _mmm_with_adapters(acp, W_basecp, out_topo, adapters)

        adapters_as_dense = torch.cat(
            [stk.ops.to_dense(adap) for adap in adapters], dim=1
        )
        adaps_as_dense = stk.ops.to_dense(adaps)
        assert (
            torch.sum(adapters_as_dense != adaps_as_dense) == 0
        ), "adapters and adaps should be the same"

        # Validate the results.
        out = stk.ops.to_dense(out)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size()[0], out.size()[0])
        self.assertEqual(expected_out.size()[1], out.size()[1])
        self.assertTrue(allclose(out, expected_out))


if __name__ == "__main__":
    unittest.main()
