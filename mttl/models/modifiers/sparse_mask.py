from dataclasses import dataclass
import os
import re
from typing import List, Union
import numpy as np
import torch
from torch import nn
import math
import os
from torch import nn
import torch
import math
import bitsandbytes as bnb
import types
import copy
from mttl.utils import logger
from scipy.sparse import csr_matrix
from mttl.models.modifiers.base import Modifier
from mttl.models.modifiers.base import (
    ModifyMixin,
    ModifierConfig,
)
import torch.nn.functional as F
from torch.autograd import Function
from spops import sddmm, csr_add


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
    M, N, BLOCK_SIZE = 4, 4, 2
    indexer = MatrixBlockIndexer(M, N, BLOCK_SIZE)
    # Example: Get indices for block 3
    block_i = 3
    indices = indexer.get_block_indices(block_i)
    indices_list = [indexer.get_block_indices(i) for i in range(indexer.L_BLOCK)]
    """

    def __init__(self, M=4, N=4, BLOCK_SIZE=2):

        if M % BLOCK_SIZE != 0 or N % BLOCK_SIZE != 0:
            raise ValueError("M  and must be divisible by BLOCK_SIZE")

        self.M = M
        self.N = N
        self.BLOCK_SIZE = BLOCK_SIZE

        self.calculate_params()

    def convert_mat_2_block(self, W_idx):
        # reshape it to get the indices of every block
        W_idx = W_idx.reshape(
            self.M // self.BLOCK_SIZE,
            self.BLOCK_SIZE,
            self.N // self.BLOCK_SIZE,
            self.BLOCK_SIZE,
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


def get_block_mask(tensor, grad, keep_ratio, block_indexer: MatrixBlockIndexer):
    """
    BLOCK-SPARSE mask calculation, returns a mask with the same dtype as the input tensor
    """
    num_params_to_keep = int(torch.numel(tensor) * keep_ratio)
    num_blocks_to_keep = int(num_params_to_keep / (block_indexer.BLOCK_SIZE**2))
    # get block scores
    block_grad = block_indexer.convert_mat_2_block(grad)
    block_score = block_grad.sum(dim=(1, 2))

    # find the top-k blocks
    threshold, topk_block_idx = torch.topk(block_score, num_blocks_to_keep, sorted=True)
    accepted_score = threshold[-1]

    # get mask-indices of the top-k blocks
    keep_masks_idx = [block_indexer.get_block_indices(i) for i in topk_block_idx]
    keep_masks_idx = torch.stack(keep_masks_idx).flatten().to(tensor.device)

    # get the mask
    keep_masks = torch.zeros_like(tensor)
    keep_masks.flatten().scatter_add_(
        0,
        keep_masks_idx,
        torch.ones(keep_masks_idx.shape, device=tensor.device, dtype=keep_masks.dtype),
    )
    return keep_masks


def get_regular_sparse_mask(tensor, grad, keep_ratio, **kwargs):
    """
    parameter-wise sparse calculation, returns a mask with the same dtype as the input tensor
    """
    num_params_to_keep = int(torch.numel(tensor) * keep_ratio)
    threshold, _ = torch.topk(grad.flatten(), num_params_to_keep, sorted=True)
    accepted_score = threshold[-1]
    keep_masks = torch.zeros_like(tensor)
    keep_masks[grad >= accepted_score] = 1.0
    assert keep_masks.dtype == tensor.dtype
    return keep_masks


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


def mod_forward(self, x):
    return torch.nn.functional.linear(x, self.weight * self.weight_mask, self.bias)


def mod_noisy_forward(self, x):
    """consider adding random noise"""
    # return torch.nn.functional.linear(x, self.weight*self.weight_mask, self.bias)
    # return torch.nn.functional.linear(x, self.weight*self.weight_mask+self.noise*self.noise_var, self.bias)
    return torch.nn.functional.linear(
        x, self.weight * self.weight_mask + self.noise, self.bias
    )
    # return torch.nn.functional.linear(x, self.weight*self.weight_mask+self.weight*self.weight_mask, self.bias)


@dataclass
class SparseMaskConfig(ModifierConfig):
    keep_ratio: float = 1.0
    BLOCK_SIZE: int = 16  # 16x
    sps_type: str = "block_sparse"  # ['block_sparse','regular_sparse']
    sps_impl: str = "sp_add+sp_mm"  # ['sp_add+sp_mm','scatter+filter']


class SparseLinearFunction_SP_ADD(Function):
    """
    Will add sparse deltas into the weights in the forward pass and calculate the gradients w.r.t. sparse weights in the backward pass.
    THis is inspired by stuff from RoSA paper and uses spops kernels.

    I guess an alternative implementation could be to use scatter add + filter out unneccessary grads in backward hook, probably like in https://arxiv.org/pdf/2401.16405 Ansel et al.

    Importantly: spops is compiled to be used with sm_80 architectures, e.g. A100. If you dont have such GPU there will be an error here, in which case you the scatter-add implementation or recompile spops to be used with your GPU.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
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
    @torch.cuda.amp.custom_bwd
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


class BlcokSparseLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, dense_weight, dense_bias, sparse_weight):
        output = F.linear(input, dense_weight, dense_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = grad_sparse_weight = None
        return grad_input, grad_weight, grad_bias, grad_sparse_weight


class SparseWeights(nn.Module):
    def __init__(self, weight, bias, config: SparseMaskConfig):
        super(SparseWeights, self).__init__()
        self.dense_layer_weight = weight.contiguous()
        self.dense_layer_bias = None
        if bias is not None:
            self.dense_layer_bias = bias.contiguous()
            self.dense_layer_bias.requires_grad = False
        self.dense_layer_weight.requires_grad = False

        self.config = config
        shape = self.dense_layer_weight.shape
        dtype = self.dense_layer_weight.dtype

        nnz = int(self.config.keep_ratio * np.prod(shape))
        self.register_buffer(
            "row_offs", torch.zeros((shape[0] + 1,), dtype=torch.int32)
        )
        self.register_buffer("row_idx", torch.zeros((shape[0],), dtype=torch.int16))
        self.register_buffer("col_idx", torch.zeros((nnz,), dtype=torch.int16))

        _sparse_csr_representation = self._init_sparse_weights()
        self.set_sparse_weights(_sparse_csr_representation)
        self.sparse_weights: nn.Parameter = nn.Parameter(
            torch.tensor(_sparse_csr_representation.data, dtype=dtype),
            requires_grad=True,
        ).contiguous()

    @torch.no_grad()
    def set_sparse_weights(self, sparse_tensor):
        self.row_offs = torch.tensor(
            sparse_tensor.indptr,
            dtype=torch.int32,
            device=self.dense_layer_weight.device,
        )
        self.col_idx = torch.tensor(
            sparse_tensor.indices,
            dtype=torch.int16,
            device=self.dense_layer_weight.device,
        )
        self.row_idx = torch.argsort(-1 * torch.diff(self.row_offs)).to(torch.int16)

    @torch.no_grad()
    def _init_sparse_weights(self):
        """
        Init sparse weights randomly. This uses CSR representaiton from scipy.
        """
        if self.config.sps_type == "block_sparse":
            block_indexer = MatrixBlockIndexer(
                M=self.dense_layer_weight.shape[0],
                N=self.dense_layer_weight.shape[1],
                BLOCK_SIZE=self.config.BLOCK_SIZE,
            )
            random_grad = torch.randn_like(self.dense_layer_weight)
            keep_params = get_block_mask(
                self.dense_layer_weight,
                random_grad,
                self.config.keep_ratio,
                block_indexer,
            )
        elif self.config.sps_type == "regular_sparse":
            random_grad = torch.randn_like(self.dense_layer_weight)
            keep_params = get_regular_sparse_mask(
                self.dense_layer_weight, random_grad, self.config.keep_ratio
            )
        else:
            raise NotImplementedError

        keep_params = keep_params.contiguous().float()
        sparse_weights = csr_matrix(keep_params.cpu())
        sparse_weights *= 0.0
        return sparse_weights

    @property
    def scipy_representation(self):
        return csr_matrix(
            (
                self.sparse_weights.cpu().data.float(),
                self.col_idx.cpu(),
                self.row_offs.cpu(),
            ),
            shape=self.dense_layer_weight.shape,
        )

    @property
    def twod_indices(self):
        val = self.sparse_weights.cpu().data.float() + 1.0
        return csr_matrix(
            (val, self.col_idx.cpu(), self.row_offs.cpu()),
            shape=self.dense_layer_weight.shape,
        ).nonzero()


class MaskedLinear(nn.Module):
    """
    Masked linear layer is used to calculate the sparse mask indices a la SNIP (https://arxiv.org/pdf/1810.02340).
    It actually calculates gradients w.r.t. a mask that is of the same size as the weight matrix but throws away grads during the backward pass.
    """

    def __init__(self, weight, bias, config: SparseMaskConfig, parent_name=None):
        super().__init__()
        self.device = weight.device
        self.weight = weight
        self.bias = bias
        self.mask = nn.Parameter(
            torch.ones(
                self.weight.shape,
                device=self.device,
                dtype=self.weight.dtype,
            )
        )

        self.parent_name = parent_name
        input_dim, output_dim = self.weight.T.shape
        self.BLOCK_SIZE = config.BLOCK_SIZE
        self.keep_ratio = config.keep_ratio
        self.config = config

        self.weight.requires_grad = False
        self.bias.requires_grad = False
        self.mask.requires_grad = True

        self.BlockwiseConvolution = MatrixBlockIndexer(
            M=input_dim, N=output_dim, BLOCK_SIZE=self.BLOCK_SIZE
        )
        self.selected_params = None

        def mask_backward_hook(mask):
            if self.config.sps_type == "regular_sparse":
                selected_params = self.get_regular_sparse_mask(mask.grad)
            elif self.config.sps_type == "block_sparse":
                selected_params = self.get_block_mask(mask.grad)
            else:
                raise NotImplementedError(
                    f"Choose `sps_type` from ['block_sparse','regular_sparse'] "
                )
            selected_params = selected_params.to_sparse_csr()
            if self.selected_params == None:
                self.selected_params = selected_params
            else:
                self.selected_params += selected_params
            mask.grad = None
            return None

        self._backward_hooks = []
        hook_handle = self.mask.register_post_accumulate_grad_hook(mask_backward_hook)
        self._backward_hooks.append(hook_handle)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight * self.mask, self.bias)

    def get_block_mask(self, mask_grad):
        """
        BLOCK-SPARSE mask calculation
        """
        return get_block_mask(
            self.weight, mask_grad, self.keep_ratio, self.BlockwiseConvolution
        )

    def get_regular_sparse_mask(self, mask_grad):
        """
        parameter-wise sparse calculation
        """
        return get_regular_sparse_mask(self.weight, mask_grad, self.keep_ratio)

    def unregister_hooks(self):
        for hook in self._backward_hooks:
            hook.remove()
        self._backward_hooks = []


class SparseLinearModule(SparseWeights):
    """
    Implements a 'real' sparse linear layer with sparse weights.
    """

    def __init__(self, weight, bias, config: SparseMaskConfig):
        super(SparseLinearModule, self).__init__(weight, bias, config)

        if self.config.sps_type == "regular_sparse":
            self.sparse_func = SparseLinearFunction_SP_ADD
        elif self.config.sps_type == "block_sparse":
            raise NotImplementedError
            self.sparse_func = BlcokSparseLinearFunction
        else:
            raise NotImplementedError

    def forward(self, input):
        return self.sparse_func.apply(
            input,
            self.dense_layer_weight,
            self.dense_layer_bias,
            self.sparse_weights,
            self.row_idx,
            self.col_idx,
            self.row_offs,
        )


class ScatteredSparseLinearModule(SparseWeights):
    """
    This implementation uses scatter-add to update the sparse weights in the forward and backward hooks to filter uneccessary gradients.
    """

    def __init__(self, weight, bias, config: SparseMaskConfig):
        super(ScatteredSparseLinearModule, self).__init__(weight, bias, config)
        
        idxs = torch.tensor(np.array(self.twod_indices),dtype=torch.int64, device=self.dense_layer_weight.device)
        self.register_buffer("idxs", idxs) # will also sync the device to the device of the model

    def forward(self, input):
        weights = self.dense_layer_weight        
        row_indices, col_indices = self.idxs[0], self.idxs[1]
        flat_indices = row_indices * weights.size(1) + col_indices
        flat_weights = weights.view(-1)
        updated_flat_weights = flat_weights.scatter_add(0, flat_indices, self.sparse_weights)
        
        weights = updated_flat_weights.view_as(weights)
        
        return torch.nn.functional.linear(input, weights, self.dense_layer_bias)


@Modifier.register("sparse_mask_adapter", config_cls=SparseMaskConfig)
class SparseMaskAdapter(Modifier, ModifyMixin):
    def __init__(
        self,
        config: SparseMaskConfig,
        layer: nn.Module,
        **kwargs,
    ):
        self.name = kwargs.get("layer_name", None)
        super().__init__()
        self.config = config
        self.masked_layer = None
        
        self.dense_layer_weight = layer.weight
        self.dense_layer_bias = layer.bias
        
        self.sparse_mask = None

        self.sps_type = config.sps_type
        assert self.sps_type in [
            "block_sparse",
            "regular_sparse",
        ], "Choose `sps_type` from ['block_sparse','regular_sparse'] "
        self.sp_impl = config.sps_impl
        assert self.sp_impl in [
            "sp_add+sp_mm",
            "scatter+filter",
        ], "Choose `sps_type` from ['sp_add+sp_mm','scatter+filter'] "

        if self.sp_impl == "sp_add+sp_mm":
            self.sparse_layer = SparseLinearModule(
                self.dense_layer_weight, self.dense_layer_bias, self.config
            )
        elif self.sp_impl == "scatter+filter":
            self.sparse_layer = ScatteredSparseLinearModule(
                self.dense_layer_weight, self.dense_layer_bias, self.config
            )
        else:
            raise NotImplementedError
    
    def forward(self, input):
        if self.masked_layer is not None:
            return self.masked_layer(input)
        output = self.sparse_layer(input)
        return output

    def on_before_mask_update(self):
        # layer is turner into MaskedLinear
        # in the masked linear we actually multiply the weight with the mask ane calculate grads w.r.t to the full mask
        self.masked_layer = MaskedLinear(
            self.dense_layer_weight,
            self.dense_layer_bias,
            self.config,
            parent_name=self.name,
        ).to(self.dense_layer_weight.device)

    def on_after_mask_update(self):
        assert isinstance(self.masked_layer, MaskedLinear)
        self.masked_layer.unregister_hooks()
        self.sparse_mask = self.masked_layer.selected_params
        assert (
            self.sparse_mask is not None and self.sparse_mask.layout == torch.sparse_csr
        )
        self.masked_layer = None
