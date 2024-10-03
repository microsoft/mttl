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

def allclose(x, y, pct=0.25):
    mask = torch.isclose(x, y, rtol=5e-2)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


blocksize = 16


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

        # out = linear_ops.scattergather_adamerge(
        #     x=X,
        #     base_act=base_act,
        #     k=k,
        #     ada_weights=ada_data,
        #     row_idxs=row_idxs,
        #     col_idxs=col_idxs_t,
        #     offsets=offsets_t,
        #     block_offsets_t=block_offsets_t,
        #     ada_block_size=blocking,
        #     sorted_expert_idxs=sorted_expert_idxs,
        #     sorted_scattered_idxs=sorted_scattered_idxs,
        #     padded_block_idxs=padded_block_idxs,
        #     gates=k_weights,
        # )
        
        
        out2 = linear_ops.scattergather_adamerge2(
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
        err_Y = torch.abs(out2 - out_dumb)
        tolerance = 1e-2
        print(err_Y.max())
        assert err_Y.max() < tolerance, "Y error too large: max %0.05f" % err_Y.max()


if __name__ == "__main__":
    unittest.main()
