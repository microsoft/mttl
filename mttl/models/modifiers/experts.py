import re

from mttl.models.adapters import ExpertContainer
from mttl.utils import logger

from projects.wiki_experts.models.routers import MULTI_EXPERT_ROUTERS


def get_selector(config, **kwargs):
    if config.expert_routing:
        if config.expert_routing not in MULTI_EXPERT_ROUTERS:
            raise ValueError(f"Cannot find selector: {config.expert_routing}")
        return MULTI_EXPERT_ROUTERS[config.expert_routing](config, **kwargs)
    else:
        return None


def _extract_identifier(string, match_on="coder"):
    """Returns a unique identifier for the "chunk" of layers sharing the
    same underlying selector
    # e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
    """
    if match_on == "finegrained":
        return string
    if match_on == "coarsegrained":
        return ""
    return string


def add_expert_to_transformer(
    transformer,
    expert_name,
    expert_config,
    expert_weights,
    action="route",
    is_default=False,
    load_only_layers=None,
    selectors={},
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

                    if type(layer) != ExpertContainer:
                        # create an expert lora container
                        expert_container = ExpertContainer(
                            expert_config,
                            transformer.task_id_container,
                            layer,
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

                    layer_num = int(expert_container.__layer_name__.split(".")[2])

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
