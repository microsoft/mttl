import torch
from scipy.sparse import csr_matrix
from torch import nn

from mttl.logging import logger
from mttl.models.modifiers.sparse_mask_config import SparseMaskConfig
from mttl.models.modifiers.sparsity.sparse_linear import MaskedLinear, SparseLinear
from mttl.models.modifiers.sparsity.sparse_utils.utils import (
    get_2d_indices_from_csr_matrix,
    get_top_k_sparcity,
    scipy_csr_to_torch_csr,
    torch_csr_to_scipy_csr,
)
from mttl.registrable import Registrable


class MaskUpdater(nn.Module, Registrable):
    def __init__(self, config: SparseMaskConfig):
        super().__init__()
        self.config = config


@MaskUpdater.register("snip", config_cls=SparseMaskConfig)
class SNIPMaskUpdater(MaskUpdater):
    """
    It is used to periodically re-calculate the sparse mask indices a la SNIP (https://arxiv.org/pdf/1810.02340).
    To recalculate the mask, it uses ONE infoming batch to estimate the importance of each parameter.

    It accumulates learned weights in a dense CPU matrix. For MaskedLinear implementation this accumulation is already done in MaskedLinear class, since sparse mask is kept in dense format.
    This accumulation is useful e.g. to make sure that the weights that have been learned in the past and are selected again are not reinitialized to 0.
    """

    def __init__(
        self, config: SparseMaskConfig, base_weights_shape, base_weights_shape_dtype
    ):
        super().__init__(config)

        self.keep_ratio = config.keep_ratio
        self.block_size = config.block_size

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

    def switch_to_mask_update_mode(self, sparse_layer: SparseLinear):
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
            if self.sparse_layer_biases is None:
                self.sparse_layer_biases = sparse_biases.detach()
            else:
                self.sparse_layer_biases += sparse_biases.detach()

        self.binary_mask = torch.ones_like(
            self.sparse_layer_weights, device=self.sparse_layer_weights.device
        )
        self.binary_mask.requires_grad = True

        def mask_backward_hook(mask):
            selected_params_dense = get_top_k_sparcity(
                mask.grad, self.config.sps_type, self.keep_ratio, self.block_size
            )
            selected_params = selected_params_dense.float().to_sparse_csr()  # .cpu()
            if self._selected_indices == None:
                self._selected_indices = selected_params  # .coalesce()
            else:
                self._selected_indices += selected_params
                self._selected_indices = self._selected_indices  # .coalesce()

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
        if isinstance(sparse_layer, MaskedLinear):
            new_weights = self.selected_indices
        else:
            # other sparse layers than MaskedLinear, do not accumulate weights
            # so its handeled here
            new_weights = self.selected_indices
            new_weights = torch_csr_to_scipy_csr(new_weights)
            r, c = get_2d_indices_from_csr_matrix(new_weights)
            new_weights *= 0.0
            new_weights[r, c] = self.accumulated_sparse_weights[r, c].float()
            new_weights = scipy_csr_to_torch_csr(new_weights)

        sparse_layer.reset_sparse_weights(new_weights)
        self._selected_indices = None
        self.binary_mask = None
        self._n_mask_updates += 1

    @property
    def selected_indices(self) -> torch.Tensor:
        if self.config.steps_in_mask_selection == 1:
            return self._selected_indices
        raise NotImplementedError(
            "More than one step in mask selection is not supported"
        )

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x = x.to(self.sparse_layer_weights.dtype)
        bias = (
            self.sparse_layer_biases.detach().to(self.sparse_layer_weights.dtype)
            if self.sparse_layer_biases is not None
            else None
        )
        assert self.sparse_layer_weights is not None
        return torch.nn.functional.linear(
            x, self.sparse_layer_weights.detach() * self.binary_mask, bias
        ).to(input_dtype)

    def unregister_hooks(self):
        for hook in self._backward_hooks:
            hook.remove()
        self._backward_hooks = []
