from dataclasses import dataclass

from mttl.models.containers.selectors.base import (
    PerTokenSelector,
    PerTokenSelectorConfig,
    Selector,
)


def compute_hidden_states(
    library,
    selector_data_id=None,
    use_base_model_only=False,
    max_samples_per_task=10,
    recompute_prototypes=False,
    track="each_layer",
    pool="last",
    default_args=None,
) -> str:
    from mttl.models.library.library_transforms import (
        HiddenStateComputer,
        HiddenStateComputerConfig,
    )

    cfg = HiddenStateComputerConfig(
        name=selector_data_id,
        use_base_model_only=use_base_model_only,
        max_samples_per_task=max_samples_per_task,
        track=track,
        pool=pool,
    )
    HiddenStateComputer(cfg).transform(
        library,
        recompute=recompute_prototypes,
        default_args=default_args,
        persist=True,
    )
    return cfg.save_name


@dataclass
class AverageActivationSelectorConfig(PerTokenSelectorConfig):
    router_temp: float = -1
    moe_top_k: int = -1
    proto_init: str = "avg_act"
    input_norm_fn: str = "id"
    proto_norm_fn: str = "id"


@Selector.register("avg_act_router", AverageActivationSelectorConfig)
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
