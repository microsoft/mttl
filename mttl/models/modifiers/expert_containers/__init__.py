import re
from mttl.config import Config
from mttl.models.modifiers.expert_containers.selectors import *
from mttl.models.modifiers.expert_containers.expert_containers import *
from mttl.utils import logger


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
    else:
        raise ValueError(f"Cannot find modifier: {modifier}")


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
        print(expert_name)
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
    expert_config,
    expert_weights,
    action="route",
    is_default=False,
    load_only_layers=None,
    selectors={},
    config=None,
):
    # create a shared container for the task id
    if not hasattr(transformer, "task_id_container"):
        transformer.task_id_container = {}

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
                        CONTAINER_CLASS = get_container_class(
                            expert_config.model_modifier
                        )
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

                    # subset the relevant expert weights starting w __layer_name__
                    subset_expert_weights = {
                        k.replace(expert_container.__layer_name__ + ".", ""): v
                        for k, v in expert_weights.items()
                        if k.startswith(expert_container.__layer_name__)
                    }

                    # get the layer number
                    pattern = r"h\.(\d+)"
                    match = re.search(pattern, expert_container.__layer_name__)
                    layer_num = match.group(1)

                    if load_only_layers:
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
                        expert_config,
                        subset_expert_weights,
                        action=action,
                        is_default=is_default,
                    )

    logger.info("Adding expert to layers %s", added_layers)
    return transformer
