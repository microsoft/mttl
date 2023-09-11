import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from dataclasses import dataclass
from typing import List
import re

from mttl.models.adapters import Adapter, ExpertContainer
from mttl.utils import logger


def add_expert_to_transformer(
    transformer,
    expert_name,
    expert_config,
    expert_weights,
    action="route",
    is_default=False,
):
    # create a shared container for the task id
    if not hasattr(transformer, "task_id_container"):
        transformer.task_id_container = {}

    total_layers = 0

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(expert_config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(expert_config.modify_layers, c_name):
                    total_layers += 1

                    if type(layer) != ExpertContainer:
                        # create an expert lora container
                        expert_container = ExpertContainer(
                            expert_config,
                            transformer.task_id_container,
                            layer,
                        )
                        expert_container.__layer_name__ = f"{m_name}.{c_name}"
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
                    expert_container.add_expert(
                        expert_name,
                        expert_config,
                        subset_expert_weights,
                        action=action,
                        is_default=is_default,
                    )
    return transformer
