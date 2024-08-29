import pytest


def test_load_selectors_from_config():
    from mttl.models.containers.selectors.base import (
        TaskNameSelector,
        TaskNameSelectorConfig,
        get_selector,
    )
    from mttl.models.containers.selectors.moe_selector import MOERKHSSelectorConfig

    config = MOERKHSSelectorConfig(emb_dim=10)

    with pytest.raises(ValueError, match="MOERKHSSelector requires a layer"):
        # raises value error due to MOERKHSSelectorConfig not having a layer
        selector = get_selector(config)

    config = TaskNameSelectorConfig()
    selector = get_selector(config)
    assert type(selector) == TaskNameSelector
