import re
from mttl.config import Config
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.expert_containers.selectors import (
    Selector,
    SelectorConfig,
    SelectorView,
    get_selector,
)
from mttl.models.modifiers.expert_containers.expert_containers import *
from mttl.models.modifiers.modify_model import CONFIGS_TO_MODIFIERS
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.expert import Expert


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


def get_container_class(modifier: str):
    import os

    if modifier == "lora":
        if os.environ.get("COALESCED_LORA_CONTAINER", "False") == "1":
            return CoalescedLoRAExpertContainer
        return LoRAExpertContainer
    elif modifier == "skilled_lora":
        if not os.environ.get("COALESCED_LORA_CONTAINER", "False") == "1":
            logger.warning(
                "COALESCED_LORA_CONTAINER is not set to 1, but still using it for SkilledLoRA"
            )
        return CoalescedLoRAExpertContainer
    elif modifier == "kv_adapter":
        return KVExpertContainer
    else:
        raise ValueError(f"Cannot find modifier: {modifier}")


def filter_expert_weights(layer_name, expert_weights):
    # subset the relevant expert weights starting w __layer_name__
    keys = list(expert_weights.keys())

    if "transformer.h" in keys[0] and "layers." in layer_name:
        # phi-huggingface to phi-private
        weights = {}
        for k, v in expert_weights.items():
            k = k.replace("transformer.h.", "layers.")
            ks = k.split(".")
            ks[1] = int(ks[1]) + 1
            k = ".".join(map(str, ks))
            weights[k] = v
    else:
        weights = expert_weights

    weights = {
        k.replace(layer_name + ".", ""): v
        for k, v in weights.items()
        if k.startswith(layer_name)
    }
    if not weights:
        return None
    return weights


def add_expert_library_to_transformer(
    transformer,
    expert_library: ExpertLibrary,
    action: str = "route",
    default_expert: str = None,
    routing_config: SelectorConfig = None,
    training_config: Config = None,
):
    for expert_name, expert_dump in expert_library.items():
        add_expert_to_transformer(
            transformer,
            expert_name,
            expert_dump.expert_config,
            expert_dump.expert_weights,
            action=action,
            is_default=expert_name == default_expert,
            routing_config=routing_config,
            training_config=training_config,
        )


def create_selector_for_container(
    transformer,
    container,
    modifier_type: str,
    selector_config: SelectorConfig,
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
        identifier not in transformer.selectors[modifier_type]
        or transformer.selectors[modifier_type].get(identifier).config
        != selector_config
    )
    if create_new_selector:
        # Special case when you have a decoder layer in an enc-dec model
        selector = get_selector(
            selector_config,
            info_container=transformer.info_container,
            layer=container.layer,
        )
        selector.__layer_name__ = identifier + ".selector"
        transformer.selectors[modifier_type][identifier] = selector

        # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
        selector.total_calls_per_forward += 1
    else:
        selector: Selector = transformer.selectors[modifier_type][identifier]
        # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
        selector.total_calls_per_forward += 1
        selector = selector.create_view()
    return selector


def replace_selector_for_container(
    transformer,
    modifier_type: str,
    selector_config: SelectorConfig,
    selector_weights: dict = None,
    force_replace: bool = False,
):
    """
    Assigns a selector to the expert containers in the transformer model.
    """
    from mttl.models.modifiers.modify_model import get_modifier_type

    expert_containers = []
    for _, module in dict(transformer.named_modules()).items():
        for _, layer in dict(module.named_children()).items():
            if isinstance(layer, ExpertContainer):
                # check if the container holds the same modifier type, e.g. LoRAConfig --> "lora"
                container_modifier = get_modifier_type(layer.__supports_configs__[0])
                # selector does not apply to this container
                if not container_modifier == modifier_type:
                    continue
                else:
                    expert_containers.append(layer)

    if not expert_containers:
        raise ValueError(
            f"No expert containers found for modifier type: {modifier_type}. Cannot assign a selector! Load some experts beforehand."
        )

    # stores the selectors per container type
    if not hasattr(transformer, "selectors"):
        transformer.selectors = {}

    if not modifier_type in transformer.selectors:
        transformer.selectors[modifier_type] = {}

    if force_replace:
        for container in expert_containers:
            container.selector = None
        transformer.selectors[modifier_type] = {}

    n_selectors = 0
    n_selectors_views = 0
    for container in expert_containers:
        selector = create_selector_for_container(
            transformer,
            container,
            modifier_type,
            selector_config,
        )
        if selector is None:
            continue

        container.assign_selector(selector)
        n_selectors += isinstance(selector, Selector)
        n_selectors_views += isinstance(selector, SelectorView)

    if selector_weights is not None:
        raise NotImplementedError()

    return n_selectors, n_selectors_views


def add_expert_to_transformer(
    transformer,
    expert: Expert,
    action: str = "route",
    is_default: bool = False,
    routing_config: SelectorConfig = None,
) -> Expert:
    """
    Routine to add an expert to the transformer architecture.

    Params:
        transformer: the transformer model to modify
        Config: the config of the model to which the expert is added
    """
    expert_config = expert.expert_config

    if not expert.name:
        raise ValueError("Expert name cannot be empty!")

    from mttl.models.modifiers.modify_model import get_modifier_type
    from mttl.models.modifiers.expert_containers.hard_prompts_container import (
        add_hard_prompt_to_transformer,
    )

    model_modifier = get_modifier_type(expert_config)

    if model_modifier == "hard_prompt":
        return add_hard_prompt_to_transformer(
            transformer,
            expert,
            action=action,
            is_default=is_default,
        )

    total_layers = 0
    added_layers = []

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(expert_config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(expert_config.modify_layers, c_name):
                    total_layers += 1
                    layer_name = f"{m_name}.{c_name}"

                    if not isinstance(layer, ExpertContainer):
                        CONTAINER_CLASS = get_container_class(model_modifier)
                        expert_container = CONTAINER_CLASS(
                            expert_config,
                            transformer.info_container,
                            layer,
                            lora_merge_after=(
                                routing_config.lora_merge_after
                                if routing_config
                                else False
                            ),
                        )
                        expert_container.__layer_name__ = layer_name
                        setattr(
                            module,
                            c_name,
                            expert_container,
                        )
                    else:
                        expert_container = layer

                    added_layers.append(expert_container.__layer_name__)
                    expert_container.add_expert(
                        expert,
                        action=action,
                        is_default=is_default,
                    )

    if routing_config is not None:
        replace_selector_for_container(
            transformer,
            model_modifier,
            routing_config,
        )

        if not transformer.selectors[model_modifier]:
            raise ValueError(
                "No selectors were created but a routing config was specified. Check your routing_config and model architecture."
            )

        logger.info(
            "Added expert %s, with %s selectors",
            expert.name,
            len(transformer.selectors[model_modifier]),
        )

    logger.debug("Patched layers: %s", added_layers)
    return expert
