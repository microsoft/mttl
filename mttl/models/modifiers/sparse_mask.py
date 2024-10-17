from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn
from triton.ops.blocksparse.matmul import dsd_lut, sdd_lut

from mttl.logging import logger
from mttl.models.modifiers.base import Modifier, ModifyMixin
from mttl.models.modifiers.sparse_utils.sparse_linear import (
    MaskedLinear,
    ScatteredSparseLinearModule,
    SparseLinear,
    SparseLinearConfig,
)
from mttl.models.modifiers.sparse_utils.utils import (
    _scatter_add_flattened,
    create_csr_tensor,
    get_2d_indices_from_csr_matrix,
    get_top_k_sparcity,
    init_sparse_weights,
    to_block_sparse_layout,
    torch_coo_to_scipy_csr,
)
from mttl.registrable import Registrable


@dataclass
class SparseMaskConfig(SparseLinearConfig):
    steps_in_mask_selection: int = (
        1  # fo how many batches stay in mask update regime where sparse weights are fixed but masks are updated
    )
    mask_reselection_interval: int = (
        100  # every how many steps to switch to mask update regime
    )
    n_max_mask_reselection: int = (
        -1
    )  # how many mask updates to do. If > 0, the mask updater will be removed after this many updates
    mask_updater: str = None  # "snip"
    remove_mask_updater_after_update_step: int = (
        None  # number of mask update steps after which the mask updater is removed and the mask is fixed for the rest of training
    )
    skip_zeros_mask_update: bool = (
        False  # if True, until the first mask update operate in full FT regime.
    )


class MaskUpdater(nn.Module, Registrable):
    def __init__(self, config: SparseMaskConfig):
        super().__init__()
        self.config = config


@MaskUpdater.register("snip", config_cls=SparseMaskConfig)
class SNIPMaskUpdater(MaskUpdater):
    """
    SNIPMaskUpdater is a wrapper around SparseLinear.
    It is used to periodically re-calculate the sparse mask indices a la SNIP (https://arxiv.org/pdf/1810.02340).
    To recalculate the mask, it uses a couple of incoming mini-batches to estimate the importance of each parameter.

    It accumulates learned weights in a dense CPU matrix. 
    This is useful e.g. to make sure that the weights that have been learned in the past and are selected again are not reinitialized to 0.
    """

    def __init__(
        self, config: SparseMaskConfig, base_weights_shape, base_weights_shape_dtype
    ):
        super().__init__(config)

        self.keep_ratio = config.keep_ratio
        self.block_size = config.block_size

        self._steps_since_last_mask_update = int(config.skip_zeros_mask_update)
        self._mask_update_steps = 0
        self._n_mask_updates = 0

        self.updating_the_mask = False

        self.binary_mask = None
        self._selected_indices = None
        self._backward_hooks = []
        self.sparse_layer_weights, self.sparse_layer_biases = None, None

        # sparse weights for accumulation on CPU
        self.accumulated_sparse_weights = torch.zeros(
            base_weights_shape, device="cpu", dtype=base_weights_shape_dtype
        )

    def switch_to_mask_update_modus(self, sparse_layer):
        self.updating_the_mask = True
        self._selected_indices = None
        base_weights, base_biases, sparse_weights, sparse_biases = (
            sparse_layer.get_weights_for_mask_learning()
        )
        if isinstance(sparse_layer, MaskedLinear):
            # here we already keep sparse weights as dense matrix, so accumulation in SNIP is not needed
            self.sparse_layer_weights = base_weights + sparse_weights
        else:
            assert isinstance(sparse_weights, csr_matrix)
            # need to do two things:
            # 1. keep track of accumulated sparse weights
            # 2. Merge those accumulated weight deltas into the base weights and use them for importance estimation
            r, c = get_2d_indices_from_csr_matrix(sparse_weights)
            if len(r) > 0:
                self.accumulated_sparse_weights[r, c] = torch.tensor(
                    sparse_weights[r, c],
                    dtype=self.accumulated_sparse_weights.dtype,
                    device="cpu",
                )
            self.sparse_layer_weights = (
                base_weights + self.accumulated_sparse_weights.to(base_weights.device)
            )

        self.sparse_layer_biases = base_biases
        if sparse_biases is not None:
            self.sparse_layer_biases += sparse_biases

        self.binary_mask = torch.ones_like(
            self.sparse_layer_weights, device=self.sparse_layer_weights.device
        )
        self.binary_mask.requires_grad = True

        def mask_backward_hook(mask):
            selected_params_dense = get_top_k_sparcity(
                mask.grad, self.config.sps_type, self.keep_ratio, self.block_size
            )
            selected_params = selected_params_dense.float().to_sparse_coo()  # .cpu()
            if self._selected_indices == None:
                self._selected_indices = selected_params.coalesce()
            else:
                self._selected_indices += selected_params
                self._selected_indices = self._selected_indices.coalesce()

            mask.grad = None  # be efficient, throw aways the grads
            return None

        hook_handle = self.binary_mask.register_post_accumulate_grad_hook(
            mask_backward_hook
        )
        self._backward_hooks.append(hook_handle)

    def switch_to_weights_update_mode(self, sparse_layer: SparseLinear):
        self.unregister_hooks()
        self.updating_the_mask = False
        self.sparse_layer_weights, self.sparse_layer_biases = None, None
        # update the mask of the sparse layer
        # SNIP weight accumulation: we set the newly selected weights to zeros,
        # but weights that have been already learned in the past are kept
        r_idx, c_idx = self.selected_indices
        csr_weights = create_csr_tensor(
            r_idx,
            c_idx,
            self.accumulated_sparse_weights[r_idx, c_idx],
            self.accumulated_sparse_weights.shape[0],
            self.accumulated_sparse_weights.shape[1],
        )
        sparse_layer.reset_sparse_weights(csr_weights)
        self._selected_indices = None
        self.binary_mask = None
        self._n_mask_updates += 1

    @property
    def selected_indices(self) -> torch.Tensor:
        if self.config.steps_in_mask_selection == 1:
            return (
                self._selected_indices.indices()[0].cpu(),
                self._selected_indices.indices()[1].cpu(),
            )
        # _selected_indices keeps track of how many times each parameter has been selected
        # an alternative, coudl be to actually accumulate gradients for the mask, but it can be too memory expensive, we coudl use cuantization.
        # Now we need to take keep_ratio of the selected params
        # since we used more than 1 batch to estimate the most important ones, some will be selected more than once and some only once
        # self._selected_indices = self._selected_indices

        selected_indices_dense = self._selected_indices.to_dense()
        selected_indices_dense = get_top_k_sparcity(
            selected_indices_dense,
            self.config.sps_type,
            self.keep_ratio,
            self.block_size,
        )
        idxs = selected_indices_dense.cpu().nonzero(as_tuple=True)
        return idxs[0], idxs[1]

    def _time_to_update_mask(self, sparse_layer: SparseLinear):
        return (
            self._steps_since_last_mask_update % self.config.mask_reselection_interval
            == 0
            and sparse_layer.training
            and self.config.n_max_mask_reselection <= self._n_mask_updates
        )

    def _time_to_update_sparse_weights(self, sparse_layer: SparseLinear):
        return (
            self._mask_update_steps % self.config.steps_in_mask_selection == 0
            and sparse_layer.training
        )

    def prepare_mask_or_weights_learning(self, sparse_layer: SparseLinear):
        """
        Currently we have two regimes that we alternate:
        - mask learning: update the non-zero indices
        - weight learning: update the sparse weights

        Here we figure out what regume we are in.
        """
        if self._time_to_update_mask(sparse_layer) and not self.updating_the_mask:
            self.switch_to_mask_update_modus(sparse_layer)
            self._mask_update_steps += 1

        elif self.updating_the_mask and not self._time_to_update_sparse_weights(
            sparse_layer
        ):
            self._mask_update_steps += 1

        elif self.updating_the_mask and self._time_to_update_sparse_weights(
            sparse_layer
        ):
            self.switch_to_weights_update_mode(sparse_layer)
            self._mask_update_steps = 0
            self._steps_since_last_mask_update = 0

        if not self.updating_the_mask:
            self._steps_since_last_mask_update += 1

    def forward(self, sparse_layer: SparseLinear, x: torch.Tensor):
        self.prepare_mask_or_weights_learning(sparse_layer)
        bias = (
            self.sparse_layer_biases.detach()
            if self.sparse_layer_biases is not None
            else None
        )
        if self.updating_the_mask:
            assert self.sparse_layer_weights is not None
            return torch.nn.functional.linear(
                x, self.sparse_layer_weights.detach() * self.binary_mask, bias
            )
        return sparse_layer(x)

    def unregister_hooks(self):
        for hook in self._backward_hooks:
            hook.remove()
        self._backward_hooks = []


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
        self.sps_type = config.sps_type
        self.sparse_layer: SparseLinear = None
        self.mask_updater: MaskUpdater = None
        if not self.config.mask_updater is None:
            self.mask_updater: MaskUpdater = MaskUpdater.get_class_by_name(
                config.mask_updater,
            )(
                self.config,
                base_weights_shape=self.dense_layer_weight.shape,
                base_weights_shape_dtype=self.dense_layer_weight.dtype,
            )

    def forward(self, input):
        if self.mask_updater is not None:
            return self.mask_updater(self.sparse_layer, input)
        return self.sparse_layer(input)


class ScatteredConfig(SparseMaskConfig):
    pass


@Modifier.register("scattered_sparse_adapter", config_cls=ScatteredConfig)
class ScatteredSparseAdapter(SparseMaskAdapter):
    """
    Sparse adapter that only keeps non-zero weights around as parameters.
    """

    def __init__(
        self,
        config: ScatteredConfig,
        layer: nn.Module,
        **kwargs,
    ):
        super().__init__(config, layer, **kwargs)
        self.sparse_layer: SparseLinear = ScatteredSparseLinearModule(
            self.dense_layer_weight,
            self.dense_layer_bias,
            self.config,
            parent_name=self.name,
        )


class MLSConfig(SparseMaskConfig):
    init_all_ones: bool = False


@Modifier.register("mls_sparse_adapter", config_cls=MLSConfig)
class MaskedLinearSparseAdapter(SparseMaskAdapter):
    """
    Sparse adapter that keeps the sparse weights as dense matrix.
    """

    def __init__(
        self,
        config: MLSConfig,
        layer: nn.Module,
        **kwargs,
    ):
        super().__init__(config, layer, **kwargs)
        self.sparse_layer: SparseLinear = MaskedLinear(
            self.dense_layer_weight,
            self.dense_layer_bias,
            self.config,
            parent_name=self.name,
            init_all_ones=config.init_all_ones,
        )
