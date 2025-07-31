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
    module, batch, parameter_selection_procedure="per_layer"
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
                m.revert_weight_grad_and_update_mask(keep_masks)


def mod_forward(self, x):
    return torch.nn.functional.linear(x, self.weight * self.weight_mask, self.bias)


@dataclass
class SparseMaskConfig(ModifierConfig):
    keep_ratio: float = 0.05
    mask_cat: str = "scatter"
    BLOCK_SIZE: int = 16  # 16x
    sparse_cat: str = "block_sparse"  # ['block_sparse','regular_sparse']
    non_trainable_param_patterns: str = "sparse_layer.weight_mask"


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
        self.sparse_layer.weight_mask = nn.Parameter(
            torch.ones(self.sparse_layer.weight.shape).to(device=layer.weight.device),
            requires_grad=False,
        )
        self.mask_cat = config.mask_cat
        self.keep_ratio = config.keep_ratio
        self.keed_mask_idx = None  # will be initialized during training

        self.patch_forward()

    def patch_forward(self):
        self.sparse_layer.forward = types.MethodType(mod_forward, self.sparse_layer)

    def data_preprocess(self, x):
        sparse_model_dtype = self.sparse_layer.weight.dtype
        return x.to(sparse_model_dtype)

    def forward(self, input):
        output = self.layer(input)

        if self.sparse_layer.weight.device != self.sparse_layer.weight_mask.device:
            raise ValueError(
                f"weight and weight_mask should be on the same device, "
                f"but got weight on {self.sparse_layer.weight.device} and "
                f"weight_mask on {self.sparse_layer.weight_mask.device}. "
                "Please check the device."
            )
        sparse_output = self.sparse_layer(
            self.data_preprocess(input)
        )  # Bfloat16-->Float32

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
                self.sparse_layer.weight_mask = nn.Parameter(
                    torch.zeros_like(self.sparse_layer.weight), requires_grad=False
                )
                self.sparse_layer.weight_mask.flatten().scatter_add_(
                    0,
                    self.keep_mask_idx,
                    torch.ones(self.keep_mask_idx.shape).to(
                        self.sparse_layer.weight.device
                    ),
                )
            else:
                self.sparse_layer.weight_mask = nn.Parameter(
                    mask, requires_grad=False
                ).to(self.sparse_layer.weight.device)
        else:
            print("Mask is not provided, initializing to default mask value=1")
            del self.sparse_layer.weight_mask
            self.sparse_layer.weight_mask = nn.Parameter(
                torch.ones(self.sparse_layer.weight.shape).to(
                    self.sparse_layer.weight.device
                ),
                requires_grad=False,
            )
