from dataclasses import dataclass
from typing import Dict
from mttl.models.library.library_transforms import (
    HiddenStateComputer,
    HiddenStateComputerConfig,
)
import torch
from mttl.models.containers.selectors import (
    PerTokenSelector,
    PerTokenSelectorConfig,
    register_multi_expert_selector,
)


def get_hidden_states(library, args):
    cfg = HiddenStateComputerConfig(
        use_base_model_only=args.use_base_model_only,
        max_samples_per_task=args.max_samples_per_task,
        name=args.expert_embeds_save_name,
        track=args.track,
        pool=args.pool,
    )
    output = HiddenStateComputer(cfg).transform(
        library, recompute=args.recompute_prototypes, default_args=args
    )

    return output


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
