import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from mttl.logging import logger

try:
    import linear_sd
except ImportError:

    logger.info(
        "linear_sd not available, LinearWithSparseDelta will not work. You can install it from https://github.com/AlanAnsell/peft"
    )


try:
    from spops import csr_add, sddmm
except ImportError:

    logger.info(
        "spops not available. You can install it with `pip install -e 'git+https://github.com/IST-DASLab/spops.git'"
    )

from torch.autograd import Function


def create_csr_tensor(row_indices, col_indices, values, num_rows, num_cols):
    """
    Constructs a PyTorch CSR tensor from given row indices, column indices, and values.

    Parameters:
    - row_indices (torch.Tensor): Tensor of row indices of non-zero values (dtype=torch.int64).
    - col_indices (torch.Tensor): Tensor of column indices of non-zero values (dtype=torch.int64).
    - values (torch.Tensor): Tensor of non-zero values.
    - num_rows (int): Number of rows in the resulting matrix.
    - num_cols (int): Number of columns in the resulting matrix.

    Returns:
    - csr_tensor (torch.Tensor): PyTorch CSR sparse tensor.
    """
    # Ensure the indices are of type torch.int64
    row_indices = row_indices.to(torch.int64)
    col_indices = col_indices.to(torch.int64)

    # Step 1: Sort the indices lexicographically by (row_indices, col_indices)
    # Stack row_indices and col_indices to form pairs
    indices = torch.stack((row_indices, col_indices), dim=1)
    # Sort indices lexicographically
    sorted_indices, sort_order = torch.sort(indices, dim=0)
    # Use the sort order to sort the values
    sorted_row_indices = row_indices[sort_order[:, 0]]
    sorted_col_indices = col_indices[sort_order[:, 1]]
    sorted_values = values[sort_order[:, 0]]

    # Step 2: Compute crow_indices
    # Count the number of non-zero entries per row
    counts = torch.bincount(sorted_row_indices, minlength=num_rows)
    # Compute the cumulative sum to get crow_indices
    crow_indices = torch.zeros(num_rows + 1, dtype=torch.int64)
    crow_indices[1:] = torch.cumsum(counts, dim=0)

    # Step 3: Create the CSR tensor
    csr_tensor = torch.sparse_csr_tensor(
        crow_indices, sorted_col_indices, sorted_values, size=(num_rows, num_cols)
    )

    return csr_tensor


def _scatter_add_flattened(weights, weights_sparse, idxs):
    """
    Adds sparse weights to the passed weights.
    Does it without in-place operations.
    """
    row_indices, col_indices = idxs[0], idxs[1]
    flat_indices = row_indices * weights.size(1) + col_indices
    # weights.flatten().scatter_add(0, flat_indices, weights_sparse)

    flat_weights = weights.view(-1)
    updated_flat_weights = flat_weights.scatter_add(0, flat_indices, weights_sparse)

    weights = updated_flat_weights.view_as(weights)
    return weights


def get_2d_indices_from_csr_matrix(sparse_tensor: csr_matrix):
    """
    Given a csr_matrix, return the row and column indices of data elements.
    We evoid just calling .nonzero() on the sparse tensor as values may be zeros.

    """
    dat = np.ones_like(sparse_tensor.data)
    return csr_matrix(
        (dat, sparse_tensor.indices, sparse_tensor.indptr), shape=sparse_tensor.shape
    ).nonzero()


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
    matrix = matrix.flatten(2, 3).sum(dim=-1)
    return matrix.cpu().bool().to(torch.int64)


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


def scipy_csr_to_torch_csr(scipy_csr_matrix):
    """
    Converts a SciPy CSR matrix to a PyTorch CSR tensor.

    Args:
        scipy_csr_matrix (scipy.sparse.csr_matrix): A SciPy CSR matrix.

    Returns:
        torch.sparse_csr_tensor: The equivalent PyTorch CSR tensor.

    Raises:
        TypeError: If the input is not a SciPy CSR matrix.
        ValueError: If the input matrix is not in CSR format.
    """
    # Validate input type
    if not isinstance(scipy_csr_matrix, csr_matrix):
        raise TypeError("Input must be a SciPy CSR matrix (scipy.sparse.csr_matrix).")

    # Ensure the matrix is in CSR format
    if not scipy_csr_matrix.format == "csr":
        raise ValueError(
            f"Input matrix must be in CSR format. Current format: {scipy_csr_matrix.format}"
        )

    # Extract CSR components from the SciPy matrix
    data = scipy_csr_matrix.data
    indices = scipy_csr_matrix.indices
    indptr = scipy_csr_matrix.indptr
    shape = scipy_csr_matrix.shape

    torch_data = torch.from_numpy(data).to(
        dtype=torch.float32
    )  # Adjust dtype as needed
    torch_indices = torch.from_numpy(indices).to(dtype=torch.int64)
    torch_indptr = torch.from_numpy(indptr).to(dtype=torch.int64)

    # Create the PyTorch CSR tensor
    torch_csr = torch.sparse_csr_tensor(
        crow_indices=torch_indptr,
        col_indices=torch_indices,
        values=torch_data,
        size=shape,
        dtype=torch_data.dtype,
    )

    return torch_csr


def torch_csr_to_scipy_csr(torch_csr_tensor):
    """
    Converts a PyTorch CSR tensor to a SciPy CSR matrix.

    Args:
        torch_csr_tensor (torch.Tensor): A PyTorch tensor with CSR layout.

    Returns:
        scipy.sparse.csr_matrix: The equivalent SciPy CSR matrix.

    Raises:
        TypeError: If the input is not a PyTorch tensor.
        ValueError: If the tensor is not in CSR layout.
    """
    # Validate input type
    if not isinstance(torch_csr_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    # Validate tensor layout
    if not torch_csr_tensor.layout == torch.sparse_csr:
        raise ValueError("Input tensor must be in CSR layout (torch.sparse_csr).")

    # Ensure tensor is on CPU
    torch_csr_cpu = torch_csr_tensor.to("cpu")

    # Extract CSR components
    crow = torch_csr_cpu.crow_indices().numpy()  # indptr
    col = torch_csr_cpu.col_indices().numpy()  # indices
    data = torch_csr_cpu.values().numpy()  # data
    shape = torch_csr_cpu.size()

    # Construct SciPy CSR matrix
    scipy_csr = csr_matrix((data, col, crow), shape=shape)

    return scipy_csr


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
    return matrix.bool().cpu().to(torch.int64)


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
    Spops is compiled to be used with sm_80 architectures, e.g. A100. If you dont have such GPU there will be an error here, in which case you the scatter-add implementation or recompile spops to be used with your GPU.
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
        # this line is from here: https://github.com/IST-DASLab/peft-rosa/blob/810da6a13f7bf3f38bc10778d589a4342351e846/src/peft/tuners/rosa/rosa_functions.py#L107
        dsW = sddmm(
            row_offs,
            row_idx,
            col_idx,
            grad_output.mT.contiguous(),
            input.mT.contiguous(),
            backend="sputnik",
        )
        return dX, None, None, dsW, None, None, None


class BlcokSparseLinearFunction_SP_ADD(Function):
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
        c_lut,
        block_size,
    ):
        # using csr_add is faster than scatter add
        sparse_weights = sparse_weights.flatten()
        weights = csr_add(sparse_weights, row_offs, row_idx, col_idx, dense_weights)
        output = F.linear(input, weights, dense_bias)
        ctx.save_for_backward(
            input,
            sparse_weights,
            row_idx,
            col_idx,
            row_offs,
            dense_weights,
            c_lut,
            block_size,
        )
        return output

    @staticmethod
    # @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        (
            input,
            sparse_weights,
            row_idx,
            col_idx,
            row_offs,
            dense_weights,
            c_lut,
            block_size,
        ) = ctx.saved_tensors
        from triton.ops.blocksparse.matmul import _matmul

        weights = csr_add(
            sparse_weights, row_offs, row_idx, col_idx, dense_weights
        )  # could be done also with torch.sparse.sampled_addmm

        block_size = block_size.item()
        spdims = (1, weights.shape[0] // block_size, weights.shape[1] // block_size)
        dX = grad_output @ weights
        grad_output = grad_output.contiguous()
        input = input.contiguous()

        dsW = _matmul.fn["sdd"](
            grad_output.unsqueeze(1),
            input.unsqueeze(1),
            True,
            False,
            False,
            spdims,
            block_size,
            c_lut,
            None,
            out=None,
        ).sum(0)

        # dsW is n_blocks x block_size x block_size
        # import pdb
        # pdb.set_trace()
        return dX, None, None, dsW, None, None, None, None, None


class BlcokSparseLinearFunction_SP_SCATTER(Function):

    @staticmethod
    # @torch.amp.custom_fwd
    def forward(
        ctx,
        input,
        dense_weights,
        dense_bias,
        sparse_weights,
        idxs,
        c_lut,
        block_size,
    ):
        # using csr_add is faster than scatter add
        sparse_weights = sparse_weights.flatten()
        weights = _scatter_add_flattened(dense_weights, sparse_weights, idxs)
        output = F.linear(input, weights, dense_bias)
        ctx.save_for_backward(
            input,
            sparse_weights,
            idxs,
            dense_weights,
            c_lut,
            block_size,
        )
        return output

    @staticmethod
    # @torch.amp.custom_bwd
    def backward(ctx, grad_output):
        (
            input,
            sparse_weights,
            idxs,
            dense_weights,
            c_lut,
            block_size,
        ) = ctx.saved_tensors
        from triton.ops.blocksparse.matmul import _matmul

        weights = _scatter_add_flattened(dense_weights, sparse_weights, idxs)
        block_size = block_size.item()
        spdims = (1, weights.shape[0] // block_size, weights.shape[1] // block_size)
        dX = grad_output @ weights
        grad_output = grad_output.contiguous()
        input = input.contiguous()

        dsW = _matmul.fn["sdd"](
            grad_output.unsqueeze(1),
            input.unsqueeze(1),
            True,
            False,
            False,
            spdims,
            block_size,
            c_lut,
            None,
            out=None,
        ).sum(0)

        # dsW is n_blocks x block_size x block_size
        # import pdb
        # pdb.set_trace()
        return dX, None, None, dsW, None, None, None


class LinearWithSparseDelta(torch.autograd.Function):
    """
    copied from https://github.com/AlanAnsell/peft
    """

    @staticmethod
    def forward(ctx, input, weight, dv, di, bias, weight_grad_hook, compute_dtype):
        ctx.save_for_backward(input, weight, dv, di, bias)
        ctx.weight_grad_hook = weight_grad_hook
        ctx.compute_dtype = compute_dtype
        # if BNB_AVAILABLE and isinstance(weight, bnb.nn.Params4bit):
        #     weight = bnb.functional.dequantize_4bit(
        #         weight,
        #         quant_state=weight.quant_state,
        #     ).to(compute_dtype)

        return linear_sd.forward(input, weight, dv, di, bias)

    @staticmethod
    def backward(ctx, output_grad):
        input, weight, dv, di, bias = ctx.saved_tensors
        if BNB_AVAILABLE and isinstance(weight, bnb.nn.Params4bit):
            weight = bnb.functional.dequantize_4bit(
                weight,
                quant_state=weight.quant_state,
            ).to(ctx.compute_dtype)

        grads = linear_sd.backward(
            output_grad,
            input,
            weight,
            dv,
            di,
            ctx.needs_input_grad[0],
            ctx.weight_grad_hook is not None or ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
            bias is not None and ctx.needs_input_grad[4],
            bias,
        )
        if ctx.weight_grad_hook is not None:
            ctx.weight_grad_hook(grads[1])

        # need to return extra values corresponding to weight_grad_hook and compute_dtype
        grads.extend([None, None])
        if ctx.needs_input_grad[1]:
            return tuple(grads)
        else:
            return (grads[0], None) + tuple(grads[2:])
