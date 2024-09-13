import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

try:
    from spops import csr_add, sddmm
except ImportError:
    from mttl.logging import logger

    logger.info(
        "spops not available. You can install it with `pip install -e 'git+https://github.com/IST-DASLab/spops.git'"
    )

from torch.autograd import Function


def torch_coo_to_scipy_csr(coo_tensor):
    """
    Convert a torch COO tensor to a scipy CSR matrix
    """
    if isinstance(coo_tensor, csr_matrix):
        return coo_tensor
    return csr_matrix(
        (
            coo_tensor.values().cpu().float(),
            (
                coo_tensor.indices()[0].cpu().long(),
                coo_tensor.indices()[1].cpu().long(),
            ),
        ),
        shape=coo_tensor.shape,
    )


def load_mask(f_name):
    mask_dict = np.load(f"{f_name}.npz", allow_pickle=True)["arr"]
    return mask_dict


def save_mask(module, f_name):
    """
    to load the saved mask use the `load_mask` function
    """
    mask_dict = {}
    import numpy as np

    for m_name, m in dict(module.named_modules()).items():
        if "dense_layer" in m_name:
            mask_dict[m_name] = torch.nonzero(m.weight_mask.data).cpu().numpy()
    # Ensure the directory exists or create it if not
    os.makedirs("saved_masks", exist_ok=True)
    np.savez_compressed(f"{f_name}.npz", arr=mask_dict)


class MatrixBlockIndexer:
    """
    Example use case:
    M, N, block_size = 4, 4, 2
    indexer = MatrixBlockIndexer(M, N, block_size)
    # Example: Get indices for block 3
    block_i = 3
    indices = indexer.get_block_indices(block_i)
    indices_list = [indexer.get_block_indices(i) for i in range(indexer.L_BLOCK)]
    """

    def __init__(self, M=4, N=4, block_size=2):

        if M % block_size != 0 or N % block_size != 0:
            raise ValueError("M  and must be divisible by block_size")

        self.M = M
        self.N = N
        self.block_size = block_size

        self.calculate_params()

    def convert_mat_2_block(self, W_idx):
        # reshape it to get the indices of every block
        W_idx = W_idx.reshape(
            self.M // self.block_size,
            self.block_size,
            self.N // self.block_size,
            self.block_size,
        )
        W_idx = W_idx.permute(0, 2, 1, 3).flatten(0, 1)
        return W_idx

    def calculate_params(self):
        # build a matrix of indices
        W_idx = torch.arange(self.M * self.N).reshape(self.M, self.N)
        # instead of storing all the indices, store the indices of one block,
        W_idx = self.convert_mat_2_block(W_idx)
        # and store the offset for every block
        self.first_block_idx = W_idx[0]
        self.block_offset = W_idx[:, 0, 0].flatten()
        self.L_BLOCK = len(self.block_offset)

    def get_block_indices(self, block_i):
        block_i_indices = self.block_offset[block_i] + self.first_block_idx
        return block_i_indices
        # get_i = lambda i : self.block_offset[i] + self.first_block_idx


def to_block_sparse_layout(matrix: torch.Tensor, block_size: int):
    """
    Returns layout of block sparse matrix: i.e. a matrix of shape (M//block_size, N//block_size) where each element is a boolean indicating whether the block is non-zero.
    """
    M, N = matrix.shape
    assert M % block_size == 0, "M must be divisible by block_size"
    assert N % block_size == 0, "N must be divisible by block_size"
    matrix = matrix.reshape(
        M // block_size,
        block_size,
        N // block_size,
        block_size,
    ).permute(0, 2, 1, 3)
    matrix = matrix.flatten(2, 3).sum(dim=2)
    return matrix.bool()


def top_k_row_sparcify(grad, keep_ratio):
    """
    ROW-SPARSE mask calculation
    Selects the top-k rows based on the sum of the absolute values of each row.

    Parameters:
    - grad: A 2D tensor where each row will be evaluated.
    - keep_ratio: A float between 0 and 1 indicating the ratio of rows to keep.

    Returns:
    - keep_masks: A mask tensor of the same shape as `grad`, where only the top-k rows are kept.
    """

    grad = torch.abs(grad)
    num_params_to_keep = int(torch.numel(grad) * keep_ratio)
    row_scores = torch.abs(grad).sum(dim=1)
    num_rows_to_keep = int(math.ceil(num_params_to_keep / grad.size(1)))

    # Find the indices of the top-k rows
    _, topk_row_idx = torch.topk(row_scores, num_rows_to_keep, sorted=True)
    keep_masks = torch.zeros_like(grad, dtype=torch.bool)
    keep_masks[topk_row_idx] = 1

    return keep_masks.to(grad.dtype)


def top_k_block_sparcify(grad, keep_ratio, block_indexer: MatrixBlockIndexer):
    """
    BLOCK-SPARSE mask calculation
    """
    grad = torch.abs(grad)
    num_params_to_keep = int(torch.numel(grad) * keep_ratio)
    num_blocks_to_keep = int(
        math.ceil(num_params_to_keep / (block_indexer.block_size**2))
    )  # round up the number of params to keep to the nearest block
    # get block scores
    block_grad = block_indexer.convert_mat_2_block(grad)
    block_score = block_grad.sum(dim=(1, 2))

    # find the top-k blocks
    threshold, topk_block_idx = torch.topk(block_score, num_blocks_to_keep, sorted=True)
    # get mask-indices of the top-k blocks
    keep_masks_idx = [block_indexer.get_block_indices(i) for i in topk_block_idx]
    keep_masks_idx = torch.stack(keep_masks_idx).flatten().to(grad.device)

    # get the mask
    len(keep_masks_idx.unique())
    keep_masks = torch.zeros_like(grad, dtype=torch.bool)
    keep_masks.flatten().scatter_add_(
        0,
        keep_masks_idx,
        torch.ones(len(keep_masks_idx), device=grad.device, dtype=torch.bool),
    )
    # note: if use bfloat in torch.ones(len(keep_masks_idx), device=grad.device, dtype=grad.dtype), then there is this weird behaviour:
    # a = torch.ones(len(keep_masks_idx), device=grad.device, dtype=torch.bfloat16)
    # torch.sum(a) < len(keep_masks_idx), weirdly...
    return keep_masks.to(grad.dtype)


def top_k_sparcify(grad, keep_ratio, **kwargs):
    """
    parameter-wise sparse calculation
    """
    grad = torch.abs(grad)
    num_params_to_keep = int(torch.numel(grad) * keep_ratio)
    _, idxs = torch.topk(grad.flatten(), num_params_to_keep, sorted=True)
    # accepted_score = threshold[-1]
    keep_masks = torch.zeros_like(grad, dtype=torch.bool)
    keep_masks.flatten().scatter_add_(
        0,
        idxs,
        torch.ones(num_params_to_keep, device=grad.device, dtype=torch.bool),
    )
    return keep_masks.to(grad.dtype)


def get_top_k_sparcity(grad, sps_type, keep_ratio, block_size=None):
    if sps_type == "regular_sparse":
        selected_params_dense = top_k_sparcify(grad, keep_ratio)
    elif sps_type == "block_sparse":
        block_indexed = MatrixBlockIndexer(
            M=grad.size(0), N=grad.size(1), block_size=block_size
        )
        selected_params_dense = top_k_block_sparcify(grad, keep_ratio, block_indexed)
    elif sps_type == "row_sparse":
        selected_params_dense = top_k_row_sparcify(grad, keep_ratio)

    else:
        raise NotImplementedError(
            f"Choose `sps_type` from ['block_sparse','regular_sparse','row_sparse'] "
        )
    return selected_params_dense


@torch.no_grad()
def init_sparse_weights(sps_type, keep_ratio, shape, block_size=None):
    """
    Init sparse weights randomly. This uses CSR representaiton from scipy.
    """
    random_grad = torch.randn(shape)
    keep_params = get_top_k_sparcity(random_grad, sps_type, keep_ratio, block_size)
    return keep_params


def make_sparse_model_during_training(module, batch):
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule

    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            m.on_before_mask_update()
    loss = module.forward(batch)
    loss.backward()
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            m.on_after_mask_update()


class SparseLinearFunction_SP_ADD(Function):
    """
    Will add sparse deltas into the weights in the forward pass and calculate the gradients w.r.t. sparse weights in the backward pass.
    THis is inspired by stuff from RoSA paper and uses spops kernels.

    I guess an alternative implementation could be to use scatter add + filter out unneccessary grads in backward hook, probably like in https://arxiv.org/pdf/2401.16405 Ansel et al.

    Importantly: spops is compiled to be used with sm_80 architectures, e.g. A100. If you dont have such GPU there will be an error here, in which case you the scatter-add implementation or recompile spops to be used with your GPU.
    """

    @staticmethod
    # @torch.amp.custom_fwd()
    def forward(
        ctx,
        input,
        dense_weights,
        dense_bias,
        sparse_weights,
        row_idx,
        col_idx,
        row_offs,
    ):
        weights = csr_add(sparse_weights, row_offs, row_idx, col_idx, dense_weights)
        output = F.linear(input, weights, dense_bias)
        ctx.save_for_backward(
            input, sparse_weights, row_idx, col_idx, row_offs, dense_weights
        )
        return output

    @staticmethod
    # @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        input, sparse_weights, row_idx, col_idx, row_offs, dense_weights = (
            ctx.saved_tensors
        )
        weights = csr_add(
            sparse_weights, row_offs, row_idx, col_idx, dense_weights
        )  # could be done also with torch.sparse.sampled_addmm
        dX = grad_output @ weights
        dsW = sddmm(
            row_offs, row_idx, col_idx, grad_output.T.contiguous(), input.T.contiguous()
        )
        return dX, None, None, dsW, None, None, None


class BlcokSparseLinearFunction_SP_ADD(Function):
    @staticmethod
    def forward(
        ctx, input, dense_weight, dense_bias, sparse_weight, row_idx, col_idx, row_offs
    ):
        output = F.linear(input, dense_weight, dense_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = grad_sparse_weight = None
        return grad_input, grad_weight, grad_bias, grad_sparse_weight

    @staticmethod
    # @torch.amp.custom_fwd
    def forward(
        ctx,
        input,
        dense_weights,
        dense_bias,
        sparse_weights,
        row_idx,
        col_idx,
        row_offs,
    ):
        weights = csr_add(sparse_weights, row_offs, row_idx, col_idx, dense_weights)
        output = F.linear(input, weights, dense_bias)
        ctx.save_for_backward(
            input, sparse_weights, row_idx, col_idx, row_offs, dense_weights
        )
        return output

    @staticmethod
    # @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        input, sparse_weights, row_idx, col_idx, row_offs, dense_weights = (
            ctx.saved_tensors
        )
        weights = csr_add(
            sparse_weights, row_offs, row_idx, col_idx, dense_weights
        )  # could be done also with torch.sparse.sampled_addmm
        dX = grad_output @ weights
        import pdb

        pdb.set_trace()
        dsW = sddmm(
            row_offs, row_idx, col_idx, grad_output.T.contiguous(), input.T.contiguous()
        )
        return dX, None, None, dsW, None, None, None
