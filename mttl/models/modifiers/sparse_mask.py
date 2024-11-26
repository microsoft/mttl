from dataclasses import dataclass

from torch import nn

from mttl.models.modifiers.base import Modifier
from mttl.models.modifiers.sm_config import SparseMaskConfig
from mttl.models.modifiers.sm_updater import MaskUpdater
from mttl.models.modifiers.sparse_utils.sparse_linear import (
    MaskedLinear,
    ScatteredSparseLinearModule,
    SparseLinear,
)


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
        self.maks_update_mode = False

    def forward(self, input):
        if self.maks_update_mode and self.training:
            return self.mask_updater(self.sparse_layer, input)
        return self.sparse_layer(input)

    def prepare_for_mask_update(self):
        if self.mask_updater is not None:
            self.mask_updater.switch_to_mask_update_mode(self.sparse_layer)
            self.maks_update_mode = True

    def prepare_for_weights_update(self):
        if self.mask_updater is not None:
            self.mask_updater.switch_to_weights_update_mode(self.sparse_layer)
            self.maks_update_mode = False


@dataclass
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


@dataclass
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
