import re
from mttl.config import Config
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.expert_containers.selectors import (
    SelectorConfig,
    get_selector,
)
from mttl.models.modifiers.expert_containers.expert_containers import *
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.module_graph import Expert


def _extract_identifier(string, match_on="finegrained"):
    """Returns a unique identifier for the "chunk" of layers sharing the
    same underlying selector

    e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
    """
    if match_on == "finegrained":
        return string.replace(".", "_")
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

    return {
        k.replace(layer_name + ".", ""): v
        for k, v in weights.items()
        if k.startswith(layer_name)
    }


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


def add_expert_to_transformer(
    transformer,
    expert: Expert,
    action: str = "route",
    is_default: bool = False,
    routing_config: SelectorConfig = None,
    training_config: Config = None,
):
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

    # create a shared container for the task id
    if not hasattr(transformer, "task_id_container"):
        transformer.task_id_container = {}
    if not hasattr(transformer, "selectors"):
        transformer.selectors = {}

    model_modifier = get_modifier_type(expert_config)

    if model_modifier == "hard_prompt":
        return add_hard_prompt_to_transformer(
            transformer,
            expert,
            action=action,
            is_default=is_default,
        )

    total_layers = 0
    n_selectors, n_selectors_views = 0, 0
    added_layers = []

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(expert_config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(expert_config.modify_layers, c_name):
                    total_layers += 1
                    layer_name = f"{m_name}.{c_name}"

                    if not isinstance(layer, ExpertContainer):
                        selector = None

                        if routing_config is not None:
                            identifier = _extract_identifier(
                                layer_name, routing_config.router_granularity
                            )

                            create_new_selector = (
                                identifier not in transformer.selectors
                            )
                            if create_new_selector:
                                # Special case when you have a decoder layer in an enc-dec model
                                selector = get_selector(
                                    routing_config,
                                    info_container=transformer.task_id_container,
                                    layer=layer,
                                    training_config=training_config,
                                )
                                selector.__layer_name__ = identifier + ".selector"
                                transformer.selectors[identifier] = selector
                                # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
                                selector.total_calls_per_forward += 1
                                n_selectors += 1
                            else:
                                selector: Selector = transformer.selectors[identifier]
                                # selector needs to know how many times it will be called per forward pass in order to be able to reset the cache
                                selector.total_calls_per_forward += 1
                                selector = selector.create_view()
                                n_selectors_views += 1

                        CONTAINER_CLASS = get_container_class(model_modifier)
                        expert_container = CONTAINER_CLASS(
                            expert_config,
                            transformer.task_id_container,
                            layer,
                            selector,
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

    if routing_config is not None and not transformer.selectors:
        raise ValueError(
            "No selectors were created but a routing config was specified. Check your routing_config and model architecture."
        )

    logger.info(
        "Added expert %s, with %s selectors", expert.name, len(transformer.selectors)
    )
    logger.debug("Patched layers: %s", added_layers)
    return transformer
