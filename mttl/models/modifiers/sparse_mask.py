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
    noise_add_ratio: float = 1.0
    noise_space_ratio: float = 1.0
    activate_noise: bool = True
    noise_cat: str = "targeted_noise"  # 'targeted_noise' or 'random_noise'
    training_mode: bool = True
    BLOCK_SIZE: int = 16  # 16x
    sparse_cat: str = (
        "block_sparse_sp_add"  # ['block_sparse_sp_add','regular_sparse_sp_add']
    )


class MaskedLinear(nn.Module):
    """
    Masked linear layer is used to calculate the sparse mask. It actually calculates gradients w.r.t. a matrix that is of the same size as the weight matrix but throws away grads during the backward pass.
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
            if self.config.sparse_cat == "regular_sparse_sp_add":
                selected_params = self.get_regular_sparse_mask(mask.grad)
            elif self.config.sparse_cat == "block_sparse_sp_add":
                selected_params = self.get_block_mask(mask.grad)
            else:
                raise NotImplementedError(
                    f"Choose `sparse_cat` from ['block_sparse','regular_sparse'] "
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


class SparseLinearFunction_SP_ADD(Function):
    """
    Will add sparse deltas into the weights in the forward pass and calculate the gradients w.r.t. sparse weights in the backward pass.
    THis is inspired by stuff from RoSA paper.

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
        # this can be performed using standard torch functions, but we chose to use spops here,
        # because for block sparse stuff we will definitely need cutom implementation, so keep it all a bit more custom.
        weights = csr_add(sparse_weights, row_offs, row_idx, col_idx, dense_weights)
        output = F.linear(input, weights, dense_bias) 
        ctx.save_for_backward(input, sparse_weights, row_idx, col_idx, row_offs, dense_weights)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        input, sparse_weights, row_idx, col_idx, row_offs, dense_weights = ctx.saved_tensors
        
        weights = csr_add(sparse_weights, row_offs, row_idx, col_idx, dense_weights) # could be done also with torch.sparse.sampled_addmm
        dX = grad_output @ weights        
        dsW = sddmm(row_offs, row_idx, col_idx, grad_output.T.contiguous(), input.T.contiguous())
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


class SparseLinearModule(nn.Module):
    def __init__(self, weight, bias, config: SparseMaskConfig):
        super(SparseLinearModule, self).__init__()
        self.dense_layer_weight = weight.contiguous()
        self.dense_layer_bias = bias.contiguous() if bias is not None else None

        self.config = config
        shape = self.dense_layer_weight.shape
        dtype = self.dense_layer_weight.dtype

        nnz = int(self.config.keep_ratio * np.prod(shape))
        self.register_buffer(
            "row_offs", torch.zeros((shape[0] + 1,), dtype=torch.int32)
        )
        self.register_buffer("row_idx", torch.zeros((shape[0],), dtype=torch.int16))
        self.register_buffer("col_idx", torch.zeros((nnz,), dtype=torch.int16))

        _sparse_representation = self._init_sparse_weights()
        self.set_sparse_weights(_sparse_representation)
        self.sparse_weights: nn.Parameter = nn.Parameter(
            torch.tensor(_sparse_representation.data, dtype=dtype), requires_grad=True
        ).contiguous()

        if self.config.sparse_cat == "regular_sparse_sp_add":
            self.sparse_func = SparseLinearFunction_SP_ADD
        elif self.config.sparse_cat == "block_sparse_sp_add":
            self.sparse_func = BlcokSparseLinearFunction
        else:
            raise NotImplementedError(
                f"Choose `sparse_cat` from ['regular_sparse_sp_add','block_sparse_sp_add'] "
            )
            # Other possible implementations are usong sp_mm, where we would seperately perform sparse matrix mult and then add the outputs.
            # for now we choose to use sp_add like in RoSA paper, since sp_add sounds more efficient than sp_mm.

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

    def _init_sparse_weights(self):
        """
        Init sparse weights randomly. This uses COO representaiton, since CSR results in illegal memory access error.
        """
        if self.config.sparse_cat == "block_sparse_sp_add":
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
        elif self.config.sparse_cat == "regular_sparse_sp_add":
            random_grad = torch.randn_like(self.dense_layer_weight)
            keep_params = get_regular_sparse_mask(
                self.dense_layer_weight, random_grad, self.config.keep_ratio
            )
        else:
            raise NotImplementedError(
                f"Choose `sparse_cat` from ['block_sparse_sp_add','regular_sparse_sp_add'] "
            )
        keep_params = keep_params.contiguous().float()
        sparse_weights = csr_matrix(keep_params.cpu())
        sparse_weights *= 0.0
        return sparse_weights

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

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.dense_layer.in_features,
            self.dense_layer.out_features,
            self.dense_layer.bias is not None,
        )


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

        self.dense_layer_weight = layer.weight
        self.dense_layer_bias = layer.bias
        self.masked_layer = None

        self.sparse_mask = None
        self.sparse_layer = SparseLinearModule(
            self.dense_layer_weight, self.dense_layer_bias, self.config
        )

        self.sparse_cat = config.sparse_cat
        assert self.sparse_cat in [
            "block_sparse_sp_add",
            "regular_sparse_sp_add",
        ], "Choose `sparse_cat` from ['block_sparse_sp_add','regular_sparse_sp_add'] "

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


def make_sparse_model(module, dm, keep_ratio=0.05):
    """
    useful for quick check and prototype, not used in the training
    """
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule

    # (1) preprocess the sparse-layers
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            m.on_before_mask_update()

    # (2) collect grads
    data_iter = iter(dm.train_dataloader())
    batch = next(data_iter)
    from mttl.models.utils import transfer_batch_to_device

    module = module.to("cuda")
    batch = transfer_batch_to_device(batch, module.device)
    loss = module.forward(batch)
    loss.backward()

    # (3) compute mask
    # (a) layer-wise
    mask_indx = []
    save_mask_indx = True
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            # m.dense_layer.weight_mask.grad
            num_params_to_keep = int(
                torch.numel(m.dense_layer.weight_mask) * keep_ratio
            )
            threshold, _ = torch.topk(
                m.dense_layer.weight_mask.grad.flatten(),
                num_params_to_keep,
                sorted=True,
            )
            accepted_score = threshold[-1]
            keep_masks = (m.dense_layer.weight_mask.grad >= accepted_score).float()
            # (4) revert back to original state
            # (a) reverse the require-grad: Turn on for `weight` and turn-off for `weight_mask`
            # (b) convert `module` back to `cpu`
            m.on_after_mask_update(keep_masks)

            # TODO: generate noise:
            m.generate_noise()

            # if save_mask_indx: mask_indx.append(torch.nonzero(keep_masks).data.cpu().numpy()) # nonzero finds the ind
    # (b) based on whole-net TODO
    module = module.to("cpu")
    # if save_mask_indx:
    #     import h5py
    #     import os
    #     # Ensure the directory exists or create it if not
    #     os.makedirs('saved_masks', exist_ok=True)
    #     f_name = f'saved_masks/{dm.config.finetune_task_name}'
    #     np.savez_compressed(f'{f_name}.npz', arr=mask_indx)
    #     # with h5py.File(f'{f_name}.h5', 'w') as f:
    #     #     f.create_dataset('data', data=mask_indx, compression='gzip')
    print("done")
