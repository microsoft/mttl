import pytest
import torch
from pytorch_lightning import seed_everything
from stk.matrix import Matrix

from mttl.models.modifiers.sparsity.sparse_utils import stk_matrix_utils as matrix_ops
from mttl.models.modifiers.sparsity.spb_moe import ops
from mttl.models.modifiers.sparsity.spb_moe.benchmark import dumb_forward

blocksize = 16

SC_MOE_TEST = {
    (4, 32, 64, 10, 2, 0.8, 16, torch.float32),
    (1024, 1024, 8192, 20, 2, 0.8, 16, torch.float32),
    (1024, 1024, 2048, 20, 2, 0.8, 16, torch.float32),
    (8, 128, 256, 10, 2, 0.8, 16, torch.float32),
}


@pytest.mark.skipif(
    torch.cuda.is_available() is False, reason="CUDA must be available for this test."
)
@pytest.mark.parametrize("bs, d, h, E, k, sparsity, blocking, dtype", SC_MOE_TEST)
def testScatteredMoE(bs, d, h, E, k, sparsity, blocking, dtype):
    seed_everything(42)
    device = "cuda"
    # print(f"Running test with bs={bs}, d={d}, h={h}, E={E}, k={k}, sparsity={sparsity}, blocking={blocking}, dtype={dtype}")
    logits = torch.randn(bs, E, dtype=dtype)
    weights = torch.softmax(logits.float(), axis=-1).to(dtype).to(device)
    X = torch.randn(bs, d, dtype=dtype, requires_grad=True).to(device)
    W = torch.randn(d, h, dtype=dtype, requires_grad=True).to(device)
    adaps = [
        matrix_ops._dense_and_sparse(d, h, sparsity, blocking, dtype) for _ in range(E)
    ]
    adaps_sparse = [adap[1] for adap in adaps]
    adaps_dense = [adap[0] for adap in adaps]
    ada_data = torch.stack([adap.data for adap in adaps_sparse], dim=0)
    row_idxs = torch.stack([adap.row_indices for adap in adaps_sparse], dim=0)
    col_idxs_t = torch.stack([adap.column_indices_t for adap in adaps_sparse], dim=0)
    offsets_t = torch.stack([adap.offsets_t for adap in adaps_sparse], dim=0)
    block_offsets_t = torch.stack(
        [adap.block_offsets_t for adap in adaps_sparse], dim=0
    )

    k_weights, expert_idxs = torch.topk(weights, k)
    sorted_expert_idxs, sorted_scattered_idxs = ops.flatten_and_sort(expert_idxs)
    padded_block_idxs, expert_offsets = ops.padded_block_indices(sorted_expert_idxs, E)

    base_act = torch.matmul(X, W)
    out2 = ops.scattergather_adamerge_opt(
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
    pytest.main([__file__])
