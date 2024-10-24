from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn
from triton.ops.blocksparse.matmul import dsd_lut, sdd_lut

from mttl.logging import logger
from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.models.modifiers.sparse_utils.utils import (
    BlcokSparseLinearFunction_SP_ADD,
    BlcokSparseLinearFunction_SP_SCATTER,
    LinearWithSparseDelta,
    SparseLinearFunction_SP_ADD,
    _scatter_add_flattened,
    get_2d_indices_from_csr_matrix,
    get_top_k_sparcity,
    init_sparse_weights,
    to_block_sparse_layout,
    torch_coo_to_scipy_csr,
)
from mttl.registrable import Registrable


@dataclass
class SparseMaskConfig(ModifierConfig):
    keep_ratio: float = 1.0
    block_size: int = 16  # e.g. 16x16
    selection_accumulation_steps: int = (
        1  # fo how many batches stay in mask update regime where sparse weights are fixed but masks are updated
    )
    reselection_steps: int = 100  # every how many steps to switch to mask update regime
    sps_type: str = "block_sparse"  # ['block_sparse','regular_sparse','row_sparse']
    sps_impl: str = "sp_add+sp_mm"  # ['sp_add+sp_mm','scattered', 'masked_linear']
    selection_algorithm: str = "rigl"
    reselection_rate_policy: str = "linear"
    mask_updater: str = None  # "snip"


class SparseLinear(ABC):
    def __init__(
        self,
        base_weight,
        base_bias,
        config: SparseMaskConfig,
        parent_name=None,
        use_sparse_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.parent_name = parent_name
        self.config = config
        self.keep_ratio = config.keep_ratio
        self.base_weight = base_weight.contiguous()
        self.base_bias = None
        if base_bias is not None:
            self.base_bias = base_bias.contiguous()
            self.base_bias.requires_grad = False
        self.base_weight.requires_grad = False

        self.sparse_bias = None
        if use_sparse_bias:
            self.sparse_bias = nn.Parameter(
                torch.zeros_like(
                    self.base_bias, dtype=self.base_bias.dtype, device=self.device
                ),
                requires_grad=True,
            )

    @property
    def device(self):
        return self.base_weight.device

    @abstractmethod
    def get_weights_for_mask_learning(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, csr_matrix, torch.Tensor]:
        """
        Returns weights that are used for updating the binary mask indices:
        e.g. can be base model weights o base model weights + accumulated sparse weights.

        In SNIP these are the weights that will ba masked to estimate param importance using the gradient of the mask.
        """
        pass

    @abstractmethod
    def reset_sparse_weights(self, mask: csr_matrix):
        """
        Resets the indices of sparse weights as well as their values if needed.
        """
        pass


class SparseWeights(nn.Module):
    """
    It implements essentially the CSR representation of the sparse weights.
    This is used to produce neccessary inputs to spops kernels.

    Keeps track of:
    - row offsets
    - row indices
    - column indices

    Weights are innitialized to all zeros.
    """

    def __init__(self, config: SparseMaskConfig, shape, dtype, device, **kwargs):
        super().__init__()

        self.shape = shape
        self.dtype = dtype

        self.sps_type = config.sps_type
        self.block_size = config.block_size
        self.keep_ratio = config.keep_ratio

        _sparse_csr_representation = self._init_sparse_weights()
        # innitialize weights to all zeros
        self.sparse_weights: nn.Parameter = nn.Parameter(
            torch.zeros(_sparse_csr_representation.data.shape, dtype=self.dtype),
            requires_grad=True,
        ).contiguous()

        nnz = int(self.keep_ratio * np.prod(self.shape))
        self.register_buffer(
            "row_offs", torch.zeros((self.shape[0] + 1,), dtype=torch.int32)
        )
        self.register_buffer(
            "row_idx", torch.zeros((self.shape[0],), dtype=torch.int16)
        )
        self.register_buffer("col_idx", torch.zeros((nnz,), dtype=torch.int16))

        self.set_sparse_idxs(_sparse_csr_representation)

    @property
    def device(self):
        return self.sparse_weights.device

    @torch.no_grad()
    def set_sparse_weights(self, sparse_tensor: csr_matrix):
        """
        Set the sparse weights to the weights of the passed csr_matrix.
        """
        assert (
            sparse_tensor.data.shape == self.sparse_weights.data.shape
        ), "Shape mismatch when resetting sparse weights"
        self.sparse_weights.data = torch.tensor(
            sparse_tensor.data, dtype=self.dtype, device=self.device
        ).contiguous()

    @torch.no_grad()
    def reset_sparse_weights(self, sparse_tensor: csr_matrix):
        self.set_sparse_idxs(sparse_tensor)
        self.set_sparse_weights(sparse_tensor)

    @torch.no_grad()
    def set_sparse_idxs(self, sparse_tensor: csr_matrix):
        self.row_offs = torch.tensor(
            sparse_tensor.indptr,
            dtype=torch.int32,
            device=self.device,
        )
        self.col_idx = torch.tensor(
            sparse_tensor.indices,
            dtype=torch.int16,
            device=self.device,
        )
        self.row_idx = torch.argsort(-1 * torch.diff(self.row_offs)).to(torch.int16)

    @torch.no_grad()
    def _init_sparse_weights(self):
        keep_params = init_sparse_weights(
            self.sps_type, self.keep_ratio, self.shape, self.block_size
        )
        keep_params = keep_params.contiguous().float()
        sparse_weights = csr_matrix(keep_params.cpu())
        return sparse_weights

    @property
    def scipy_representation(self):
        return csr_matrix(
            (
                self.sparse_weights.cpu().data.float(),
                self.col_idx.cpu(),
                self.row_offs.cpu(),
            ),
            shape=self.shape,
        )

    @property
    def twoD_indices(self):
        """
        Returns a simple 2d representation of the sparse weights instead of the CSR format.
        """
        return get_2d_indices_from_csr_matrix(self.scipy_representation)

    def to_dense(self):
        """
        Returns dense representation of the sparse weights.
        """
        return torch.tensor(
            self.scipy_representation.toarray(), device=self.device, dtype=self.dtype
        )

    @classmethod
    def from_dense(cls, dense_tensor: torch.Tensor, config: SparseMaskConfig):
        """
        Initialize the sparse weights from a dense tensor.
        """
        sparse_weights = cls(
            config, dense_tensor.shape, dense_tensor.dtype, dense_tensor.device
        )
        scipu_csr = torch_coo_to_scipy_csr(dense_tensor.data.to_sparse_coo())
        sparse_weights.set_sparse_idxs(scipu_csr)
        sparse_weights.set_sparse_weights(scipu_csr)
        return sparse_weights


class BlockSparseWeights(SparseWeights):
    """
    Here the sparse weights are stored as [nb-blocks x block-size x block-size] view instead of flattened.
    This is usefull for block-sparse kernels.
    """

    def __init__(self, config: SparseMaskConfig, shape, dtype, device, **kwargs):
        super().__init__(config, shape, dtype, device, **kwargs)
        self.sparse_weights = nn.Parameter(
            self.sparse_weights.data.view(-1, self.block_size, self.block_size),
            requires_grad=True,
        )

    @property
    def scipy_representation(self):
        return csr_matrix(
            (
                self.sparse_weights.flatten().cpu().data.float(),
                self.col_idx.cpu(),
                self.row_offs.cpu(),
            ),
            shape=self.shape,
        )

    @torch.no_grad()
    def set_sparse_weights(self, sparse_tensor: csr_matrix):
        """
        Set the sparse weights to the weights of the passed csr_matrix.
        """
        assert (
            sparse_tensor.data.shape == self.sparse_weights.data.flatten().shape
        ), "Shape mismatch when resetting sparse weights"
        self.sparse_weights.data = (
            torch.tensor(sparse_tensor.data, dtype=self.dtype, device=self.device)
            .view_as(self.sparse_weights)
            .contiguous()
        )


class MaskedLinear(SparseLinear, nn.Module):
    """
    A dummy method to learn the sparse weights as it operates only with dense matricies. It will keep sparse weights as dense matrix (size of the original weights) and calculate grads w.r.t. the sparse weights.

    Importantly: this accumulates sparse weights! So every time the mask is reset, it may select weights that have been adapter in the past and will not be zero.
    """

    def __init__(
        self,
        base_weight,
        base_bias,
        config: SparseMaskConfig,
        parent_name=None,
        use_sparse_bias=False,
        mask: torch.Tensor = None,
    ):
        super().__init__(base_weight, base_bias, config, parent_name, use_sparse_bias)

        self.block_size = config.block_size
        self.keep_ratio = config.keep_ratio

        binary_mask = init_sparse_weights(
            self.config.sps_type,
            self.keep_ratio,
            self.base_weight.shape,
            self.block_size,
        )
        self.binary_mask = binary_mask.to(self.device)
        self.sparse_weights = nn.Parameter(
            torch.zeros_like(
                self.base_weight, dtype=self.base_weight.dtype, device=self.device
            ),
            requires_grad=True,
        )

    def forward(self, x):
        base_out = torch.nn.functional.linear(x, self.base_weight, self.base_bias)
        self.binary_mask = self.binary_mask.to(self.device).to(self.base_weight.dtype)
        sparse_out = torch.nn.functional.linear(
            x, self.sparse_weights * self.binary_mask, self.sparse_bias
        )
        return base_out + sparse_out

    def get_weights_for_mask_learning(self):
        return (
            self.base_weight,
            self.base_bias,
            csr_matrix(
                self.sparse_weights.data.cpu().float(), shape=self.sparse_weights.shape
            ),
            self.sparse_bias,
        )

    def reset_sparse_weights(self, mask: csr_matrix):
        self.binary_mask = torch.tensor(
            mask.toarray(), device=self.base_weight.device, dtype=self.base_weight.dtype
        )
        r, c = get_2d_indices_from_csr_matrix(mask)
        self.sparse_weights.data[r, c] = torch.tensor(
            mask.data, dtype=self.base_weight.dtype, device=self.base_weight.device
        )


class SparseLinearModule(SparseWeights, SparseLinear):
    """
    Implements a sparse linear layer with sparse weights and sparse backprop.
    """

    def __init__(
        self,
        weight,
        bias,
        config: SparseMaskConfig,
        parent_name=None,
        use_sparse_bias=False,
        sparse_func=None,
    ):
        SparseWeights.__init__(self, config, weight.shape, weight.dtype, weight.device)
        SparseLinear.__init__(self, weight, bias, config, parent_name, use_sparse_bias)
        self.sparse_func = sparse_func
        if self.sparse_func is None:
            if self.config.sps_type in ["regular_sparse", "row_sparse"]:
                self.sparse_func = SparseLinearFunction_SP_ADD
            elif self.config.sps_type == "block_sparse":
                logger.warning(
                    "SparseLinearModule is not optimized for block_sparse, try using BlockSparseLinearModule"
                )
                self.sparse_func = SparseLinearFunction_SP_ADD
            else:
                raise NotImplementedError

    def forward(self, input):
        bias = self.base_bias
        if self.sparse_bias is not None:
            bias = self.base_bias + self.sparse_bias
        return self.sparse_func.apply(
            input,
            self.base_weight,
            bias,
            self.sparse_weights,
            self.row_idx,
            self.col_idx,
            self.row_offs,
        )

    def get_weights_for_mask_learning(self):
        return (
            self.base_weight,
            self.base_bias,
            self.scipy_representation,
            self.sparse_bias,
        )


class BlockSparseLinearModule(BlockSparseWeights, SparseLinear):
    """
    Implements a block sparse linear layer with block sparse weights and sparse backprop.
    """

    def __init__(
        self,
        weight,
        bias,
        config: SparseMaskConfig,
        parent_name=None,
        use_sparse_bias=False,
        sparse_func=None,
    ):
        assert (
            config.sps_type == "block_sparse"
        ), "BlockSparseLinearModule only supports block_sparse type"
        BlockSparseWeights.__init__(
            self, config, weight.shape, weight.dtype, weight.device
        )
        SparseLinear.__init__(self, weight, bias, config, parent_name, use_sparse_bias)
        self.sparse_func = sparse_func
        self.sparse_func = BlcokSparseLinearFunction_SP_ADD
        if self.sps_type != "block_sparse":
            logger.warning(
                f"Using 'triton_block_sparse' which only suppots block_sparse type but got {self.sps_type}"
            )
            self.sps_type = "block_sparse"

        layout = self.get_layout().unsqueeze(0)
        c_lut, _ = sdd_lut(layout, self.block_size, self.device)
        self.register_buffer("c_lut", c_lut)

    def get_layout(self):
        """
        Returns layout of block sparse matrix: i.e. a matrix of shape (M//block_size, N//block_size) where each element is a boolean indicating whether the block is non-zero.
        """
        sparse_weights = torch.ones_like(self.sparse_weights.flatten())
        sp_m = csr_matrix(
            (
                sparse_weights.cpu().data.float(),
                self.col_idx.cpu(),
                self.row_offs.cpu(),
            ),
            shape=self.shape,
        )
        w = torch.tensor(sp_m.toarray(), device="cpu", dtype=self.dtype)
        return to_block_sparse_layout(w, self.block_size)

    def forward(self, input):
        bias = self.base_bias
        if self.sparse_bias is not None:
            bias = self.base_bias + self.sparse_bias
        return self.sparse_func.apply(
            input,
            self.base_weight,
            bias,
            self.sparse_weights,
            self.row_idx,
            self.col_idx,
            self.row_offs,
            self.c_lut,
            torch.tensor(self.block_size),
        )

    def get_weights_for_mask_learning(self):
        return (
            self.base_weight,
            self.base_bias,
            self.scipy_representation,
            self.sparse_bias,
        )


class BlockSparseLinearModuleScatter(BlockSparseLinearModule):
    """
    Implements a block sparse linear layer with block sparse weights and sparse backprop.
    """

    def __init__(
        self,
        weight,
        bias,
        config: SparseMaskConfig,
        parent_name=None,
        use_sparse_bias=False,
        sparse_func=None,
    ):
        super().__init__(
            weight, bias, config, parent_name, use_sparse_bias, sparse_func
        )
        self.sparse_func = BlcokSparseLinearFunction_SP_SCATTER

        idxs = torch.tensor(
            np.array(self.twoD_indices),
            dtype=torch.int64,
            device=self.base_weight.device,
        )
        self.register_buffer(
            "idxs", idxs
        )  # will also sync the device to the device of the model

    def forward(self, input):
        bias = self.base_bias
        if self.sparse_bias is not None:
            bias = self.base_bias + self.sparse_bias
        return self.sparse_func.apply(
            input,
            self.base_weight,
            bias,
            self.sparse_weights,
            self.idxs,
            self.c_lut,
            torch.tensor(self.block_size),
        )

    def get_weights_for_mask_learning(self):
        return (
            self.base_weight,
            self.base_bias,
            self.scipy_representation,
            self.sparse_bias,
        )


class ScatteredSparseLinearModule(SparseWeights, SparseLinear):
    """
    This implementation uses scatter-add to update the sparse weights in the forward pass.
    The autograd should be only storing the grads wrt to the sparse weights.
    """

    def __init__(
        self,
        weight,
        bias,
        config: SparseMaskConfig,
        parent_name=None,
        use_sparse_bias=False,
        mask: torch.Tensor = None,
    ):

        SparseWeights.__init__(self, config, weight.shape, weight.dtype, weight.device)
        SparseLinear.__init__(self, weight, bias, config, parent_name, use_sparse_bias)

        idxs = torch.tensor(
            np.array(self.twoD_indices),
            dtype=torch.int64,
            device=self.base_weight.device,
        )
        self.register_buffer(
            "idxs", idxs
        )  # will also sync the device to the device of the model

    def forward(self, input):
        weights = _scatter_add_flattened(
            self.base_weight, self.sparse_weights, self.idxs
        )
        bias = self.base_bias
        if self.sparse_bias is not None:
            bias = self.base_bias + self.sparse_bias
        return torch.nn.functional.linear(input, weights, bias)

    def get_weights_for_mask_learning(self):
        return (
            self.base_weight,
            self.base_bias,
            self.scipy_representation,
            self.sparse_bias,
        )

    def reset_sparse_weights(self, mask: csr_matrix):
        SparseWeights.reset_sparse_weights(self, mask)
        self.idxs.data = torch.tensor(
            np.array(self.twoD_indices),
            dtype=torch.int64,
            device=self.base_weight.device,
        )


class SpieLSparseLinearModule(SparseLinearModule):
    """
    This implements the SpIEL kernel: https://arxiv.org/pdf/2401.16405
    """

    def __init__(
        self,
        weight,
        bias,
        config: SparseMaskConfig,
        parent_name=None,
        use_sparse_bias=False,
        mask: torch.Tensor = None,
    ):
        super().__init__(
            weight,
            bias,
            config,
            parent_name,
            use_sparse_bias,
            sparse_func=LinearWithSparseDelta,
        )
        indices = torch.tensor(
            np.array(self.oneD_indices),
            dtype=torch.int64,
            device=self.base_weight.device,
        )
        self.register_buffer("idxs", indices)

    @property
    def oneD_indices(self):
        """
        Returns a simple 1d representation of the sparse weights instead of the CSR format.
        """
        twoD_indices = self.twoD_indices
        return twoD_indices[0] * self.shape[1] + twoD_indices[1]

    def forward(self, input):
        bias = self.base_bias
        if self.sparse_bias is not None:
            bias = self.base_bias + self.sparse_bias
        return self.sparse_func.apply(
            input,
            self.base_weight,
            self.sparse_weights,
            self.idxs,
            bias,
            None,
            self.base_weight.dtype,
        )


class MaskUpdatWrapper(nn.Module, Registrable):
    def __init__(self, sparse_layer: SparseLinear, config: SparseMaskConfig):
        super().__init__()
        self.config = config
        self.sparse_layer: SparseLinear = sparse_layer


@MaskUpdatWrapper.register("spiel", config_cls=SparseMaskConfig)
class SpielMaskUpdateWrapper(MaskUpdatWrapper):
    """
    Mask update as in https://arxiv.org/pdf/2401.16405
    """

    def __init__(self, sparse_layer: SparseLinear, config: SparseMaskConfig):
        super().__init__(sparse_layer, config)

        self._steps_since_last_mask_update = 0
        self._mask_update_steps = 0
        self.sparse_layer_weights, self.sparse_layer_biases = None, None

    @property
    def _time_to_update_mask(self):
        return (
            self._steps_since_last_mask_update % self.config.reselection_steps == 0
            and self.sparse_layer.training
        )

    @property
    def _time_to_update_sparse_weights(self):
        return (
            self._mask_update_steps % self.config.selection_accumulation_steps == 0
            and self.sparse_layer.training
        )


@MaskUpdatWrapper.register("snip", config_cls=SparseMaskConfig)
class SNIPMaskUpdateWrapper(MaskUpdatWrapper):
    """
    SNIPMaskUpdateWrapper is a wrapper around SparseLinear.
    It is used to periodically re-calculate the sparse mask indices a la SNIP (https://arxiv.org/pdf/1810.02340).
    To recalculate the mask, it uses a couple of incoming mini-batches to estimate the importance of each parameter.

    This method willl also keep track of the selected weights throughout learning on CPU.

    """

    def __init__(self, sparse_layer: SparseLinear, config: SparseMaskConfig):
        super().__init__(sparse_layer, config)

        self.keep_ratio = config.keep_ratio
        self.block_size = config.block_size

        self._steps_since_last_mask_update = 0
        self._mask_update_steps = 0

        self.updating_the_mask = False

        self.binary_mask = None
        self._selected_indices = None
        self._backward_hooks = []
        self.sparse_layer_weights, self.sparse_layer_biases = None, None

        # sparse weights for accumulation on CPU
        self.accumulated_sparse_weights = torch.zeros_like(
            sparse_layer.base_weight, device="cpu"
        )

    def switch_to_mask_update_modus(self):
        self.updating_the_mask = True
        self._selected_indices = None
        base_weights, base_biases, sparse_weights, sparse_biases = (
            self.sparse_layer.get_weights_for_mask_learning()
        )
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
        self.sparse_layer_weights = base_weights + self.accumulated_sparse_weights.to(
            base_weights.device
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

    def switch_to_weights_update_modus(self):
        self.unregister_hooks()
        self.updating_the_mask = False
        self.sparse_layer_weights, self.sparse_layer_biases = None, None
        # update the mask of the sparse layer
        # SNIP sets the new weights to zeros but weights that have been learned in the past are kept
        new_weights = torch_coo_to_scipy_csr(self.selected_params)
        r, c = get_2d_indices_from_csr_matrix(new_weights)
        new_weights *= 0.0
        new_weights[r, c] = self.accumulated_sparse_weights[r, c].float()
        self.sparse_layer.reset_sparse_weights(new_weights)
        self._selected_indices = None
        self.binary_mask = None

    @property
    def selected_params(self) -> torch.Tensor:
        if self.config.selection_accumulation_steps == 1:
            return self._selected_indices
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
        return csr_matrix(selected_indices_dense.cpu())

    @property
    def _time_to_update_mask(self):
        return (
            self._steps_since_last_mask_update % self.config.reselection_steps == 0
            and self.sparse_layer.training
        )

    @property
    def _time_to_update_sparse_weights(self):
        return (
            self._mask_update_steps % self.config.selection_accumulation_steps == 0
            and self.sparse_layer.training
        )

    def prepate_mask_or_weights_learning(self):
        """
        Currently we have two regimes that we alternate:
        - mask learning: update the non-zero indices
        - weight learning: update the sparse weights

        Here we figure out what regume we are in.
        """
        if self._time_to_update_mask and not self.updating_the_mask:
            self.switch_to_mask_update_modus()
            self._mask_update_steps += 1

        elif self.updating_the_mask and not self._time_to_update_sparse_weights:
            self._mask_update_steps += 1

        elif self.updating_the_mask and self._time_to_update_sparse_weights:
            self.switch_to_weights_update_modus()
            self._mask_update_steps = 0
            self._steps_since_last_mask_update = 0

        if not self.updating_the_mask:
            self._steps_since_last_mask_update += 1

    def forward(self, x):
        self.prepate_mask_or_weights_learning()
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
        return self.sparse_layer(x)

    def unregister_hooks(self):
        for hook in self._backward_hooks:
            hook.remove()
        self._backward_hooks = []


@Modifier.register("sparse_mask_adapter", config_cls=SparseMaskConfig)
class SparseMaskAdapter(Modifier):
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
        assert self.sps_type in [
            "block_sparse",
            "regular_sparse",
            "row_sparse",
        ], "Choose `sps_type` from ['block_sparse','regular_sparse','row_sparse'] "
        self.sp_impl = config.sps_impl

        if self.sp_impl == "sp_add+sp_mm":
            sparse_layer: SparseLinear = SparseLinearModule(
                self.dense_layer_weight, self.dense_layer_bias, self.config
            )
        elif self.sp_impl == "spiel":
            sparse_layer: SparseLinear = SpieLSparseLinearModule(
                self.dense_layer_weight, self.dense_layer_bias, self.config
            )
        elif self.sp_impl == "triton_block_sparse":
            sparse_layer: SparseLinear = BlockSparseLinearModule(
                self.dense_layer_weight, self.dense_layer_bias, self.config
            )
        elif self.sp_impl == "triton_block_sparse_scatter":
            sparse_layer: SparseLinear = BlockSparseLinearModuleScatter(
                self.dense_layer_weight, self.dense_layer_bias, self.config
            )
        elif self.sp_impl == "scattered":
            sparse_layer: SparseLinear = ScatteredSparseLinearModule(
                self.dense_layer_weight, self.dense_layer_bias, self.config
            )
        elif self.sp_impl == "masked_linear":
            # dummy implementation for profiling purposes
            sparse_layer: SparseLinear = MaskedLinear(
                self.dense_layer_weight,
                self.dense_layer_bias,
                self.config,
                parent_name=self.name,
            )
        else:
            raise NotImplementedError

        # wrap sparse layer into mask updater
        if self.config.mask_updater is None:
            self.sparse_layer = sparse_layer
            # logger.warning(
            #     "No mask updater is used, using the sparse layer directly with random maks."
            # )
        else:
            self.sparse_layer: MaskUpdatWrapper = MaskUpdatWrapper.get_class_by_name(
                config.mask_updater
            )(sparse_layer, self.config)

    def forward(self, input):
        return self.sparse_layer(input)
