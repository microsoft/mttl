import copy
import json
import math
import os
import random
import re
import types
from dataclasses import dataclass
from typing import List, Union

import bitsandbytes as bnb
import numpy as np
import torch
from torch import nn

from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.utils import logger


def load_mask(f_name):
    destination_type, _ = f_name.split("://")
    if destination_type == "hf":
        from huggingface_hub import hf_hub_download

        destination_type, f_name = f_name.split("://")
        repo_id = ("/").join(f_name.split("/")[:2])
        task_name = f_name.split("/")[-1]
        f_path = hf_hub_download(repo_id=repo_id, filename=task_name)
        mask_dict = np.load(f_path, allow_pickle=True)["arr"]
    elif destination_type == "local":
        mask_dict = np.load(f"{f_name}.npz", allow_pickle=True)["arr"]
    return mask_dict


def save_mask(module, f_name):
    """
    to load the saved mask use the `load_mask` function
    """
    mask_dict = {}
    import numpy as np

    for m_name, m in dict(module.named_modules()).items():
        if "sparse_layer" in m_name:
            mask_dict[m_name] = torch.nonzero(m.weight_mask.data).cpu().numpy()
    destination_type = f_name.split("://")[0]
    # save in local dir
    if destination_type == "local":
        destination_type, f_name = f_name.split("://")
        np.savez_compressed(f"./{f_name}.npz", arr=mask_dict)

    # upload to hf
    elif destination_type == "hf":
        from huggingface_hub.hf_api import upload_file as hf_upload_file

        destination_type, f_name = f_name.split("://")
        repo_id = ("/").join(f_name.split("/")[:2])
        task_name = f_name.split("/")[-1]
        path_in_repo = f"{task_name}.npz"
        os.makedirs("./temp/test_library/", exist_ok=True)
        local_file_path = f"./temp/test_library/{path_in_repo}"
        np.savez_compressed(local_file_path, arr=mask_dict)

        hf_upload_file(
            path_or_fileobj=local_file_path,  # path saved in local machine
            path_in_repo=path_in_repo,  # path with in repo
            repo_id=repo_id,
        )
    # exact local dir is provided
    else:
        # Ensure the directory exists or create it if not
        filedir = os.path.dirname(f_name)
        os.makedirs(filedir, exist_ok=True)
        np.savez_compressed(f"./{f_name}.npz", arr=mask_dict)


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


def get_block_mask(m):
    """
    BLOCK-SPARSE mask calculation
    """
    num_params_to_keep = int(torch.numel(m.sparse_layer.weight_mask) * m.keep_ratio)
    num_blocks_to_keep = int(
        num_params_to_keep / (m.BlockwiseConvolution.BLOCK_SIZE**2)
    )
    # get block scores
    block_grad = m.BlockwiseConvolution.convert_mat_2_block(
        m.sparse_layer.weight_mask.grad
    )
    block_score = block_grad.sum(dim=(1, 2))

    # find the top-k blocks
    threshold, topk_block_idx = torch.topk(block_score, num_blocks_to_keep, sorted=True)
    accepted_score = threshold[-1]

    # get mask-indices of the top-k blocks
    keep_masks_idx = [
        m.BlockwiseConvolution.get_block_indices(i) for i in topk_block_idx
    ]
    keep_masks_idx = torch.stack(keep_masks_idx).flatten().to(m.layer.weight.device)

    # get the mask
    keep_masks = torch.zeros_like(m.sparse_layer.weight)
    keep_masks.flatten().scatter_add_(
        0,
        keep_masks_idx,
        torch.ones(
            keep_masks_idx.shape, dtype=torch.float32, device=m.layer.weight.device
        ),
    )

    return keep_masks


def get_regular_sparse_mask(m):
    """
    parameter-wise sparse calculation
    """
    num_params_to_keep = int(torch.numel(m.sparse_layer.weight_mask) * m.keep_ratio)
    threshold, _ = torch.topk(
        m.sparse_layer.weight_mask.grad.flatten(), num_params_to_keep, sorted=True
    )
    accepted_score = threshold[-1]
    keep_masks = (m.sparse_layer.weight_mask.grad >= accepted_score).float()

    return keep_masks


def make_sparse_model_during_training(
    module, batch, print_statement=False, parameter_selection_procedure="per_layer"
):
    from mttl.models.modifiers.sparse_mask import SparseMaskAdapter as SparseMaskModule

    # (1) preprocess the sparse-layers
    for m in module.modules():
        if isinstance(m, SparseMaskModule):
            m.preprocess_for_mask_update()

    # (2) collect grads
    from mttl.models.utils import transfer_batch_to_device

    loss = module.forward(**batch).loss
    loss.backward()

    assert parameter_selection_procedure in [
        "model",
        "per_layer",
    ], "choose the right `parameter_selection_procedure`"

    # (3) compute mask
    # (a) layer-wise
    if parameter_selection_procedure == "per_layer":
        for m in module.modules():
            if isinstance(m, SparseMaskModule):
                if m.sparse_cat == "block_sparse":
                    keep_masks = get_block_mask(m)
                elif m.sparse_cat == "regular_sparse":
                    keep_masks = get_regular_sparse_mask(m)

                # (4) revert back to original state
                # (a) reverse the require-grad: Turn on for `weight` and turn-off for `weight_mask`
                # (b) convert `module` back to `cpu`
                m.revert_weight_grad_and_update_mask(keep_masks)

    # (b) based on whole-net
    # b.1 compute score
    elif parameter_selection_procedure == "model":
        num_params_to_keep = 0
        grads = []
        for m in module.modules():
            if isinstance(m, SparseMaskModule):
                assert (
                    m.sparse_cat == "regular_sparse"
                ), "parameter_selection_procedure over `model` is not implemented for `block_sparse`"
                num_params_to_keep += int(
                    torch.numel(m.sparse_layer.weight_mask) * m.keep_ratio
                )
                grads.append(m.sparse_layer.weight_mask.grad.flatten().cpu())

        threshold, _ = torch.topk(
            torch.stack(grads).flatten(), num_params_to_keep, sorted=True
        )
        accepted_score = threshold[-1]
        # b.2 mask
        for m in module.modules():
            if isinstance(m, SparseMaskModule):
                keep_masks = (m.sparse_layer.weight_mask.grad >= accepted_score).float()
                if print_statement:
                    print(
                        "sparsity",
                        (keep_masks.sum() / m.sparse_layer.weight_mask.numel()) * 100,
                        "expected",
                        m.keep_ratio * 100,
                    )
                m.revert_weight_grad_and_update_mask(keep_masks)


def mod_forward(self, x):
    return torch.nn.functional.linear(x, self.weight * self.weight_mask, self.bias)


@dataclass
class SparseMaskConfig(ModifierConfig):
    keep_ratio: float = 0.05
    mask_cat: str = "scatter"
    BLOCK_SIZE: int = 16  # 16x
    sparse_cat: str = "block_sparse"  # ['block_sparse','regular_sparse']


@Modifier.register("sparse_mask_adapter", config_cls=SparseMaskConfig)
class SparseMaskAdapter(Modifier):
    def __init__(
        self,
        config: SparseMaskConfig,
        layer: nn.Module,
        **kwargs,
    ):
        super().__init__()

        self.layer = layer
        input_dim, output_dim = self.layer.in_features, self.layer.out_features
        self.param_shape = self.layer.weight.shape
        self.param_num = self.layer.weight.numel()

        self.sparse_cat = config.sparse_cat
        assert self.sparse_cat in [
            "block_sparse",
            "regular_sparse",
        ], "Choose `sparse_cat` from ['block_sparse','regular_sparse'] "

        # weight initialization
        self.sparse_layer = nn.Linear(input_dim, output_dim).to(
            device=layer.weight.device
        )
        self.sparse_layer.weight = nn.Parameter(
            torch.zeros(self.sparse_layer.weight.shape)
        )
        self.sparse_layer.bias = nn.Parameter(torch.zeros(self.sparse_layer.bias.shape))

        if self.sparse_cat == "block_sparse":
            self.BLOCK_SIZE = config.BLOCK_SIZE
            self.BlockwiseConvolution = MatrixBlockIndexer(
                M=input_dim, N=output_dim, BLOCK_SIZE=self.BLOCK_SIZE
            )

        # mask initialization
        self.sparse_layer.weight_mask = torch.ones(self.sparse_layer.weight.shape).to(
            device=layer.weight.device
        )
        self.mask_cat = config.mask_cat
        self.keep_ratio = config.keep_ratio
        self.keed_mask_idx = None  # will be initialized during training

        self.patch_forward()

    def patch_forward(self):
        self.sparse_layer.forward = types.MethodType(mod_forward, self.sparse_layer)

    @torch.no_grad()
    def convert_sparse_weight_to_1D(self):
        assert len(self.sparse_layer.weight.shape) == 2, print(
            "sparse_layer.weight is already converted to 1D"
        )
        self.sparse_layer.weight = nn.Parameter(
            self.sparse_layer.weight.flatten()[self.keep_mask_idx].data
        ).to(self.layer.weight.device)

    def data_preprocess(self, x):
        sparse_model_dtype = self.sparse_layer.weight.dtype
        return x.to(sparse_model_dtype)

    def confirm_sync_device(self):
        with torch.no_grad():
            # sync the device of the `weight_mask` with module layers
            if self.sparse_layer.weight_mask.requires_grad:
                # when gradient graph is true
                if self.sparse_layer.weight.device != self.layer.weight.device:
                    self.sparse_layer.weight_mask = self.sparse_layer.weight_mask.to(
                        self.layer.weight.device
                    )
            else:
                # otherwise
                # required `.clone()` before device transfer, otherwise getting following error
                # """RuntimeError: Inference tensors cannot be saved for backward.
                #    To work around you can make a clone to get a normal tensor and use it in autograd."""
                self.sparse_layer.weight_mask = (
                    self.sparse_layer.weight_mask.clone().to(self.layer.weight.device)
                )

    def forward(self, input):
        # `weight_mask` requires to sync with `weight` device, as default only looks module
        self.confirm_sync_device()  # TODO, need to remove this, it should only require once before training
        output = self.layer(input)

        if self.sparse_layer.weight.device != self.sparse_layer.weight_mask.device:
            print(self.sparse_layer.weight.device, self.sparse_layer.weight_mask.device)
        try:
            sparse_output = self.sparse_layer(
                self.data_preprocess(input)
            )  # Bfloat16-->Float32

        except:
            print(self.sparse_layer.weight.device, self.sparse_layer.weight_mask.device)
        return output + sparse_output.to(input.dtype)  # Float32-->Bfloat16

    """
    - prepare the mask and corresponding weight for mask-gradient calculation step during mask update
    - in this step, we want to compute weight "only" w.r.t. `weight_mask`
    """

    def preprocess_for_mask_update(self):
        # Turn off the gradient for weight
        self.sparse_layer.weight.requires_grad = False
        # init the mask
        self.sparse_layer.weight_mask = nn.Parameter(
            torch.ones(
                self.sparse_layer.weight_mask.shape, device=self.layer.weight.device
            )
        )
        # compute gradient for weight_mask
        self.sparse_layer.weight_mask.requires_grad = True

    """
    after configuring the mask, it's important to update and allow gradient to pass through the weight for training
    """

    def revert_weight_grad_and_update_mask(self, mask=None):
        # Turn back on the gradient for weight
        self.sparse_layer.weight.requires_grad = True
        # update mask
        if mask != None:
            del self.sparse_layer.weight_mask
            if self.mask_cat == "scatter":
                self.keep_mask_idx = torch.where(mask.flatten() == 1)[0].to(
                    self.sparse_layer.weight.device
                )
                self.sparse_layer.weight_mask = torch.zeros_like(
                    self.sparse_layer.weight
                )
                self.sparse_layer.weight_mask.flatten().scatter_add_(
                    0,
                    self.keep_mask_idx,
                    torch.ones(self.keep_mask_idx.shape).to(
                        self.sparse_layer.weight.device
                    ),
                )
            else:
                self.sparse_layer.weight_mask = mask.to(self.sparse_layer.weight.device)
        else:
            print("Mask is not provided, initializing to default mask value=1")
            del self.sparse_layer.weight_mask
            self.sparse_layer.weight_mask = torch.ones(
                self.sparse_layer.weight_mask.shape
            ).to(self.sparse_layer.weight.device)
