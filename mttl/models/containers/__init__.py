import re
from typing import Tuple, Type

from mttl.logging import logger, warn_once
from mttl.models.containers.base import ExpertContainer
from mttl.models.containers.lora_containers import (
    LoRAExpertContainer,
    SkilledLoRAExpertContainer,
)
from mttl.models.containers.peer_container import PEERMLPContainer
from mttl.models.containers.selectors.base import (
    Selector,
    SelectorConfig,
    SelectorsCache,
    SelectorView,
    get_selector,
)
from mttl.models.containers.utils import create_modif_regex, match_modules_to_modify
from mttl.models.library.expert import Expert
from mttl.models.modifiers.base import Modifier
from mttl.utils import logger


def _extract_identifier(string, match_on="finegrained"):
    """Returns a unique identifier for the "chunk" of layers sharing the
    same underlying selector

    e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
    """
    if match_on == "finegrained" or match_on == "*":
        return string
    if match_on == "coarsegrained":
        return "shared"
    pos = string.find(f"{match_on}")
    if pos == -1:
        raise ValueError(
            "Cannot resolve the selected router granularity: %s" % match_on
        )
    return string[: pos + len(match_on)]


def get_default_container_class(modifier_name: str) -> Type["ExpertContainer"]:
    import os

    defaults = {
        "lora": LoRAExpertContainer,
        "skilled_lora": SkilledLoRAExpertContainer,
        "peer": PEERMLPContainer,
    }
    if modifier_name not in defaults:
        raise ValueError(f"Cannot find modifier: {modifier_name}")
    return defaults[modifier_name]


def create_selector_for_container(
    transformer,
    container,
    modifier_name: str,
    selector_config: SelectorConfig = None,
    selector_cache: SelectorsCache = None,
) -> Selector:
    if container.selector is not None and container.selector.config == selector_config:
        # selector already exists and has the same config
        return

    identifier = _extract_identifier(
        container.layer_name, selector_config.router_granularity
    )

    # we create a new selector if it doesn't exist for this identifier, or
    # if we are replacing a previous one of a different type
    create_new_selector = (
        not selector_cache.get(modifier_name, identifier)
        or selector_cache.get(modifier_name, identifier).config != selector_config
    )
    if create_new_selector:
        # Special case when you have a decoder layer in an enc-dec model
        selector = get_selector(
            selector_config,
            layer=container.layer,
        )
        selector_cache.insert(modifier_name, identifier, selector)

        # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
        selector.total_calls_per_forward += 1
    else:
        selector: Selector = selector_cache.get(modifier_name, identifier)
        # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
        selector.total_calls_per_forward += 1
        selector = selector.create_view()

    # assign this selector to the container, propagates expert informations, etc.
    container.assign_selector(selector)
    return selector


def replace_selector_for_container(
    transformer,
    modifier_name: str,
    selector_config: SelectorConfig = None,
    selector_cache: SelectorsCache = None,
    force_replace: bool = False,
) -> Tuple[int, int]:
    """
    Assigns a selector to the expert containers in the transformer model.
    """
    expert_containers = []
    for _, module in dict(transformer.named_modules()).items():
        if isinstance(module, ExpertContainer):
            # check if the container holds the same modifier type, e.g. PEERConfig --> "peers"
            for supports_config in module.__supports_configs__:
                container_modifier = Modifier.get_name_by_config_class(supports_config)
                # selector does not apply to this container
                if not container_modifier == modifier_name:
                    continue
                else:
                    expert_containers.append(module)

        for _, layer in dict(module.named_children()).items():
            if isinstance(layer, ExpertContainer):
                # check if the container holds the same modifier type, e.g. LoRAConfig --> "lora"
                for supports_config in layer.__supports_configs__:
                    container_modifier = Modifier.get_name_by_config_class(
                        supports_config
                    )
                    # selector does not apply to this container
                    if not container_modifier == modifier_name:
                        continue
                    else:
                        expert_containers.append(layer)
                        break

    if not expert_containers:
        raise ValueError(
            f"No expert containers found for modifier type: {modifier_name}. Cannot assign a selector! Load some experts beforehand."
        )

    if force_replace:
        for container in expert_containers:
            container.selector = None
        selector_cache.clear(modifier_name)

    n_selectors = 0
    n_selectors_views = 0

    for container in expert_containers:
        selector = create_selector_for_container(
            transformer,
            container,
            modifier_name,
            selector_config,
            selector_cache,
        )
        n_selectors += isinstance(selector, Selector)
        n_selectors_views += isinstance(selector, SelectorView)

    return n_selectors, n_selectors_views
