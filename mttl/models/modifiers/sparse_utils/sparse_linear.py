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
from mttl.models.modifiers.base import Modifier, ModifierConfig, ModifyMixin
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
class SparseLinearConfig(ModifierConfig):
    keep_ratio: float = 1.0
    block_size: int = 16  # e.g. 16x16\]
    sps_type: str = "block_sparse"  # ['block_sparse','regular_sparse','row_sparse']
    use_sparse_bias: bool = True


class SparseLinear(ABC):
    def __init__(
        self,
        base_weight,
        base_bias,
        config: SparseLinearConfig,
        parent_name=None,
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
        if config.use_sparse_bias:
            self.sparse_bias = nn.Parameter(
                torch.zeros(self.base_weight.shape[0], dtype=self.base_weight.dtype),
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
    def reset_sparse_weights(self, mask: torch.Tensor):
        """
        Resets the indices of sparse weights as well as their values if needed.
        Mask is CSR matrix.
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

    def __init__(self, config: SparseLinearConfig, shape, dtype, device, **kwargs):
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
    def set_sparse_weights(self, sparse_tensor: Union[csr_matrix, torch.Tensor]):
        """
        Set the sparse weights to the weights of the passed csr_matrix.
        """
        data = (
            sparse_tensor.data
            if isinstance(sparse_tensor, csr_matrix)
            else sparse_tensor
        )

        assert (
            data.shape == self.sparse_weights.data.shape
        ), "Shape mismatch when resetting sparse weights"
        self.sparse_weights.data = torch.tensor(
            data, dtype=self.dtype, device=self.device
        ).contiguous()

    @torch.no_grad()
    def reset_sparse_weights(self, sparse_tensor: torch.Tensor):
        self.set_sparse_idxs(sparse_tensor)
        self.set_sparse_weights(sparse_tensor.values())

    @torch.no_grad()
    def set_sparse_idxs(self, sparse_tensor: Union[csr_matrix, torch.Tensor]):
        if isinstance(sparse_tensor, csr_matrix):
            col_idxs, row_offs = sparse_tensor.indices, sparse_tensor.indptr
        else:
            col_idxs, row_offs = (
                sparse_tensor.col_indices(),
                sparse_tensor.crow_indices(),
            )

        self.row_offs = torch.tensor(
            row_offs,
            dtype=torch.int32,
            device=self.device,
        )
        self.col_idx = torch.tensor(
            col_idxs,
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
    def from_dense(cls, dense_tensor: torch.Tensor, config: SparseLinearConfig):
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

    def __init__(self, config: SparseLinearConfig, shape, dtype, device, **kwargs):
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
    A dummy method to learn the sparse weights as it operates only with dense matricies.
    It will keep sparse weights as dense matrix (size of the original weights) and calculate grads w.r.t. the sparse weights.

    Importantly: this accumulates sparse weights! So every time the mask is reset, it may select weights that have been adapter in the past and will not be zero.
    Importantly: untill the first mask update, the mask is all ones so equivalent to the dense layer.
    """

    def __init__(
        self,
        base_weight,
        base_bias,
        config: SparseLinearConfig,
        parent_name=None,
        init_all_ones=False,
        mask: torch.Tensor = None,
    ):
        super().__init__(base_weight, base_bias, config, parent_name)

        self.block_size = config.block_size
        self.keep_ratio = config.keep_ratio
        if init_all_ones:
            binary_mask = torch.ones_like(
                self.base_weight, dtype=self.base_weight.dtype
            )
        else:
            binary_mask = init_sparse_weights(
                self.config.sps_type,
                self.keep_ratio,
                self.base_weight.shape,
                self.block_size,
            )
            binary_mask = binary_mask.to(self.device)
        self.sparse_weights = nn.Parameter(
            torch.zeros_like(
                self.base_weight, dtype=self.base_weight.dtype, device=self.device
            ),
            requires_grad=True,
        )
        self.register_buffer("binary_mask", binary_mask)

    def forward(self, x):
        base_out = torch.nn.functional.linear(x, self.base_weight, self.base_bias)
        # self.binary_mask = self.binary_mask.to(self.device).to(self.base_weight.dtype)
        sparse_out = torch.nn.functional.linear(
            x, self.sparse_weights * self.binary_mask, self.sparse_bias
        )
        return base_out + sparse_out

    def get_weights_for_mask_learning(self):
        return (
            self.base_weight,
            self.base_bias,
            # csr_matrix(
            #     self.sparse_weights.data.cpu().float(), shape=self.sparse_weights.shape
            # ),
            self.sparse_weights,
            self.sparse_bias,
        )

    def reset_sparse_weights(self, mask: torch.Tensor):
        """
        Only resets the binary mask, weights for this adapter are never reset.
        """
        mask.values().copy_(torch.ones_like(mask.values()))  # ake sure its binary
        self.binary_mask = torch.tensor(
            mask.to_dense(),
            device=self.base_weight.device,
            dtype=self.base_weight.dtype,
        )

    @property
    def scipy_representation(self):
        non_zero_indices = torch.nonzero(self.binary_mask, as_tuple=True)
        row_idx = non_zero_indices[0].numpy()
        col_idx = non_zero_indices[1].numpy()
        data = self.sparse_weights.data[row_idx, col_idx].cpu().float().numpy()
        return csr_matrix((data, (row_idx, col_idx)), shape=self.base_weight.shape)


class SparseLinearModule(SparseWeights, SparseLinear):
    """
    Implements a sparse linear layer with sparse weights and sparse backprop.
    """

    def __init__(
        self,
        weight,
        bias,
        config: SparseLinearConfig,
        parent_name=None,
        sparse_func=None,
    ):
        SparseWeights.__init__(self, config, weight.shape, weight.dtype, weight.device)
        SparseLinear.__init__(self, weight, bias, config, parent_name)
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
        config: SparseLinearConfig,
        parent_name=None,
        sparse_func=None,
    ):
        assert (
            config.sps_type == "block_sparse"
        ), "BlockSparseLinearModule only supports block_sparse type"
        BlockSparseWeights.__init__(
            self, config, weight.shape, weight.dtype, weight.device
        )
        SparseLinear.__init__(self, weight, bias, config, parent_name)
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
        config: SparseLinearConfig,
        parent_name=None,
        sparse_func=None,
    ):
        super().__init__(weight, bias, config, parent_name, sparse_func)
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
        config: SparseLinearConfig,
        parent_name=None,
    ):

        SparseWeights.__init__(self, config, weight.shape, weight.dtype, weight.device)
        SparseLinear.__init__(self, weight, bias, config, parent_name)

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

    def reset_sparse_weights(self, mask: torch.Tensor):
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
        config: SparseLinearConfig,
        parent_name=None,
        mask: torch.Tensor = None,
    ):
        super().__init__(
            weight,
            bias,
            config,
            parent_name,
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
