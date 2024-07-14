from dataclasses import dataclass
from typing import Dict
import torch
from mttl.models.containers.selectors import (
    PerTokenSelector,
    PerTokenSelectorConfig,
    register_multi_expert_selector,
)


@dataclass
class AverageActivationSelectorConfig(PerTokenSelectorConfig):
    router_temp: float = -1
    moe_top_k: int = -1
    proto_init: str = "avg_act"
    input_norm_fn: str = "id"
    proto_norm_fn: str = "id"


@register_multi_expert_selector("avg_act_router", AverageActivationSelectorConfig)
class AverageActivationSelector(PerTokenSelector):
    def _load_from_library(self):
        """Fetches prototypes from the library."""
        from mttl.models.library.library_transforms import (
            HiddenStateComputer,
            HiddenStateComputerConfig,
        )

        return HiddenStateComputer(
            HiddenStateComputerConfig(name=self.config.selector_data_id)
        ).fetch(self.config.library_id)
