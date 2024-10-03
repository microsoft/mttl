import itertools
import os
import unittest

import numpy as np
import stk
import torch
import torch.nn.functional as F
from absl.testing import parameterized
from pytorch_lightning import seed_everything
from stk.matrix import Matrix

from mttl.models.modifiers.spasity.stk import functions, linear_ops, matrix_ops

# os.environ["TRITON_INTERPRET"] = "1"


# os.environ["TRITON_INTERPRET"] = "1"


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


SC_MOE_TEST = {
    (1024, 1024, 8192, 20, 2, 0.8, 16, torch.float32),
    (1024, 1024, 2048, 20, 2, 0.8, 16, torch.float32),
    (8, 128, 256, 10, 2, 0.8, 16, torch.float32),
}

def dumb_forward(base_act, x, expert_p, expert_idxs, adaps):
    output = torch.stack(
        [
            sum(
                base_act[i]
                + (expert_p[i, j] * torch.matmul(x[i], (adaps[expert_idxs[i, j]])))
                for j in range(expert_idxs.size(1))
            )
            for i in range(expert_idxs.size(0))
        ],
        dim=0,
    )
    return output


@parameterized.parameters(*SC_MOE_TEST)
class ScatteredMoETest(parameterized.TestCase):
    def testScatteredMoE(self, bs, d, h, E, k, sparsity, blocking, dtype):
        torch.manual_seed(42)
        # print(f"Running test with bs={bs}, d={d}, h={h}, E={E}, k={k}, sparsity={sparsity}, blocking={blocking}, dtype={dtype}")
        logits = torch.randn(bs, E, dtype=dtype)
        weights = torch.softmax(logits.float(), axis=-1).cuda().to(dtype)
        X = torch.randn(bs, d, dtype=dtype, requires_grad=True).cuda()
        W = torch.randn(d, h, dtype=dtype, requires_grad=True).cuda()
        adaps = [_dense_and_sparse(d, h, sparsity, blocking, dtype) for _ in range(E)]
        adaps_sparse = [adap[1] for adap in adaps]
        adaps_dense = [adap[0] for adap in adaps]
        ada_data = torch.stack([adap.data for adap in adaps_sparse], dim=0)
        row_idxs = torch.stack([adap.row_indices for adap in adaps_sparse], dim=0)
        col_idxs_t = torch.stack(
            [adap.column_indices_t for adap in adaps_sparse], dim=0
        )
        offsets_t = torch.stack([adap.offsets_t for adap in adaps_sparse], dim=0)
        block_offsets_t = torch.stack(
            [adap.block_offsets_t for adap in adaps_sparse], dim=0
        )

        k_weights, expert_idxs = torch.topk(weights, k)
        sorted_expert_idxs, sorted_scattered_idxs = linear_ops.flatten_and_sort(
            expert_idxs
        )
        padded_block_idxs, expert_offsets = linear_ops.padded_block_indices(
            sorted_expert_idxs, E
        )

        base_act = torch.matmul(X, W)

        out = functions.parallel_linear(
            x=X,
            base_act=base_act,
            k=k,
            ada_weights=ada_data,
            row_idxs=row_idxs,
            col_idxs=col_idxs_t,
            offsets=offsets_t,
            block_offsets_t=block_offsets_t,
            ada_block_size=blocking,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            gates=k_weights,
        )
        
        
        out2 = functions.parallel_linear_optimized(
            x=X,
            base_act=base_act,
            k=k,
            ada_weights=ada_data,
            row_idxs=row_idxs,
            col_idxs=col_idxs_t,
            offsets=offsets_t,
            block_offsets_t=block_offsets_t,
            ada_block_size=blocking,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            gates=k_weights,
        )
        
        
        
        out_dumb = dumb_forward(base_act, X, k_weights, expert_idxs, adaps_dense)
        err_Y = torch.abs(out - out_dumb)
        tolerance = 1e-2
        # print(err_Y.max())
        assert err_Y.max() < tolerance, "Y error too large: max %0.05f" % err_Y.max()


if __name__ == "__main__":
    unittest.main()
