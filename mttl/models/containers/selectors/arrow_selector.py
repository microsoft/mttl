from dataclasses import dataclass

from mttl.models.containers.selectors.base import Selector, artifacts_cache
from mttl.models.containers.selectors.per_token_selector import (
    PerTokenSelector,
    PerTokenSelectorConfig,
)


def compute_arrow_embeddings(
    library,
    selector_data_id=None,
    ab_only=True,
    tie_params=None,
    tie_op="concat",
    add_base_proto=False,
    recompute_prototypes=False,
) -> str:
    from mttl.models.library.library_transforms import ArrowConfig, ArrowTransform

    cfg = ArrowConfig(
        name=selector_data_id,
        ab_only=ab_only,
        tie_params=tie_params or "default",
        tie_op=tie_op,
        add_base_proto=add_base_proto,
    )
    ArrowTransform(cfg).transform(
        library,
        recompute=recompute_prototypes,
        persist=True,
    )
    return cfg.save_name


@dataclass
class ArrowSelectorConfig(PerTokenSelectorConfig):
    router_temp: float = 1.0
    top_k: int = -1
    proto_init: str = "arrow"
    input_norm_fn: str = "id"
    proto_norm_fn: str = "id"


@Selector.register("arrow_router", ArrowSelectorConfig)
class ArrowSelector(PerTokenSelector):
    @classmethod
    @artifacts_cache
    def load_from_library(cls, config):
        """Fetches prototypes from the library."""
        from mttl.models.library.library_transforms import ArrowConfig, ArrowTransform

        return ArrowTransform.fetch(config.library_id, config.selector_data_id)
