from functools import partial
import re
from mttl.config import Config
from mttl.models.modifiers.expert_containers.selectors import *
from mttl.models.modifiers.expert_containers.expert_containers import *
from mttl.models.modifiers.kv_adapter import KVAdapter
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.module_graph import Expert


def get_selector(config: Config, **kwargs):
    if config.router_selector:
        if config.router_selector not in MULTI_EXPERT_ROUTERS:
            raise ValueError(f"Cannot find selector: {config.router_selector}")
        return MULTI_EXPERT_ROUTERS[config.router_selector](config, **kwargs)
    else:
        return None


def _extract_identifier(string, match_on="coder"):
    """Returns a unique identifier for the "chunk" of layers sharing the
    same underlying selector
    e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
    """
    assert match_on in [
        "coarsegrained",
        "finegrained",
    ], "For expert router only coarsegrained and finegrained are supported"

    if match_on == "finegrained":
        return string.replace(".", "_")
    if match_on == "coarsegrained":
        return "shared"
    return string


def get_container_class(modifier: str):
    if modifier == "lora":
        return LoRAExpertContainer
    elif modifier == "kv_adapter":
        return KVExpertContainer
    else:
        raise ValueError(f"Cannot find modifier: {modifier}")


def filter_expert_weights(layer_name, expert_weights):
    # subset the relevant expert weights starting w __layer_name__
    return {
        k.replace(layer_name + ".", ""): v
        for k, v in expert_weights.items()
        if k.startswith(layer_name)
    }


def add_expert_library_to_transformer(
    transformer,
    expert_library,
    action="route",
    default_expert=None,
    load_only_layers=None,
    selectors={},
    config=None,
):
    for expert_name, expert_dump in expert_library.items():
        add_expert_to_transformer(
            transformer,
            expert_name,
            expert_dump.expert_config,
            expert_dump.expert_weights,
            action=action,
            is_default=expert_name == default_expert,
            load_only_layers=load_only_layers,
            selectors=selectors,
            config=config,
        )


def add_expert_to_transformer(
    transformer,
    expert_name,
    expert: Expert,
    action="route",
    is_default=False,
    load_only_layers=None,
    selectors={},
    config=None,
):
    """
    Params:
        transformer: the transformer model to modify
        Config: the config of the model to which the expert is added
    """

    expert_config = expert.expert_config

    from mttl.models.modifiers.modify_model import get_modifier_type
    from mttl.models.modifiers.expert_containers.hard_prompts_container import (
        add_hard_prompt_to_transformer,
    )

    # create a shared container for the task id
    if not hasattr(transformer, "task_id_container"):
        transformer.task_id_container = {}

    model_modifier = get_modifier_type(expert_config)

    if model_modifier == "hard_prompt":
        return add_hard_prompt_to_transformer(
            transformer,
            expert_name,
            expert,
            action=action,
            is_default=is_default,
            selectors=selectors,
            config=config,
        )

    total_layers = 0
    added_layers = []

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(expert_config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(expert_config.modify_layers, c_name):
                    total_layers += 1
                    layer_name = f"{m_name}.{c_name}"

                    selector = None
                    if config is not None:
                        identifier = _extract_identifier(
                            layer_name, config.router_granularity
                        )
                        if identifier not in selectors.keys():
                            selectors[identifier] = get_selector(
                                config, info_container=transformer.task_id_container
                            )
                            if config.router_selector:
                                selectors[identifier].__layer_name__ = (
                                    identifier + ".selector"
                                )
                        selector = selectors[identifier]

                    if not isinstance(layer, ExpertContainer):
                        # create an expert lora container
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

                    if load_only_layers:
                        layer_num = int(expert_container.__layer_name__.split(".")[2])

                        pos = load_only_layers.find("-")
                        sel = int(load_only_layers.replace("-", ""))

                        if pos == 0:
                            # add until layer number excluded
                            if layer_num >= sel:
                                continue
                        else:
                            if layer_num < sel:
                                continue

                    added_layers.append(expert_container.__layer_name__)
                    expert_container.add_expert(
                        expert_name,
                        expert,
                        expert_weights=filter_expert_weights(
                            expert_container.__layer_name__, expert.expert_weights
                        ),
                        action=action,
                        is_default=is_default,
                    )

    logger.debug("Adding expert to layers %s", added_layers)
    return transformer
