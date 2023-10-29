import math
import torch
import torch.nn as nn
from enum import Enum

from mttl.models.adapters import Adapter, MergableAdapter
from typing import Any, Dict
import torch.nn.functional as F
from mttl.models.modifiers.experts import add_expert_to_transformer
from mttl.models.adapters import SkilledLoRA, LoRA, SkilledLoRA_MergeLoraAfterOP
from abc import abstractmethod, ABCMeta

MULTI_EXPERT_ROUTERS = {}


def register_multi_expert_selector(name):
    print("Registering multi-expert selector..." + name)

    def _thunk(fn):
        if name in MULTI_EXPERT_ROUTERS:
            raise ValueError(f"Cannot register duplicate model modifier ({name})")
        MULTI_EXPERT_ROUTERS[name] = fn
        return fn

    return _thunk


class GlobalRouter:
    @abstractmethod
    def init(self, expert_names, **kwargs):
        pass


@register_multi_expert_selector("local_softmax")
class Local_Multi_ExpertRouter(torch.nn.Module):
    """
    Implements routing at the level of the full model: the routing weights shared accross all layers
    """

    # TODO: implement more complex versions of this router: one that ha smore params, that takes also the inputs ad routes, etc.
    def __init__(self, config, expert_names):
        super().__init__()
        self.config = config
        self.activtion = torch.nn.Softmax(dim=-1)
        self.expert_names = expert_names
        self._merging_weights = torch.nn.parameter.Parameter(
            torch.ones(len(expert_names)), requires_grad=True
        )
        # init _merging_weights to be gaussian
        self._merging_weights.data.normal_(mean=0.0, std=0.02)
        self._is_initialized = True

    def forward(self, input, **kwargs):
        weights = self.activtion(self._merging_weights)
        return {k: v for k, v in zip(self.expert_names, weights)}


@register_multi_expert_selector("global_softmax")
class Global_Multi_ExpertRouter(torch.nn.Module, GlobalRouter):
    """
    Implements routing at the level of the full model: the routing weights shared accross all layers
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_initialized = False
        self.activtion = torch.nn.Softmax(dim=-1)

    def init(self, expert_names):
        self.expert_names = expert_names
        self._merging_weights = torch.nn.parameter.Parameter(
            torch.ones(len(expert_names)), requires_grad=True
        )
        # init _merging_weights to be gaussian
        self._merging_weights.data.normal_(mean=0.0, std=0.02)
        self._is_initialized = True

    def forward(self):
        assert self._is_initialized, "Router has not been initialized yet!"
        weights = self.activtion(self._merging_weights)
        return {k: v for k, v in zip(self.expert_names, weights)}


class RouterExpertContainer(Adapter, MergableAdapter):
    def __init__(
        self,
        config,
        task_id_container,
        layer,
    ):
        super().__init__()
        self.config = config
        self.layer = layer

        if not isinstance(self.layer, nn.Linear):
            raise ValueError(
                "Expert containers for layers other than nn.Linear have not been implemented."
            )

        self.info_container = task_id_container
        self.default_expert_name = None
        self.merged_expert_names = []
        self.experts = nn.ModuleDict({})

    def add_expert(
        self,
        name: str,
        expert_config: Any,
        expert_weights: Dict[str, torch.Tensor],
        action="merge",
        is_default=False,
    ) -> None:
        if name in self.experts:
            raise ValueError("An expert with name {} already exists.".format(name))

        if is_default and action == "merge":
            raise ValueError(
                "Cannot set is_default if this expert is merged, change to 'route'."
            )

        # hack this for now, but build a proper config for each module
        if expert_config.model_modifier == "lora":
            expert_module = LoRA(expert_config, self.layer)
            expert_module.load_lora_weights(expert_weights)
        else:
            raise NotImplementedError("ExpertContainer only supports LoRA experts.")

        if action == "merge":
            # weight is merged with layer so we can discard it now
            if expert_config.model_modifier == "lora":
                expert_module.merge_with_layer()
                self.merged_expert_names.append(name)
            else:
                raise NotImplementedError("Merging experts only supports LoRA experts.")
        else:
            # we keep track of the expert weights
            self.experts[name] = expert_module
        if is_default:
            self.default_expert_name = name

    def merge_with_layer(self):
        assert (
            len(self.experts) == 0
        ), "Cannot proceed with merging experts. Probably because some experts were added with action 'route'."
        return

    def forward(self, input, **kwargs):
        task_names = self.info_container["routing_infos"].task_names

        if task_names and (
            any(task_name not in self.experts for task_name in task_names)
            and not self.default_expert_name
            and len(self.experts)
        ):
            raise ValueError(
                "Experts for all tasks have not been loaded! Set a default expert?"
            )

        # if it has some routing experts *and* task names, then we can route
        ##################
        #### routing #####
        if len(self.experts) and task_names is not None:
            load_experts = []

            for task_name in task_names:
                if task_name not in self.experts:
                    if not self.default_expert_name:
                        raise ValueError(
                            "The expert for this task {} does not exists. Consider setting a default expert!".format(
                                task_name
                            )
                        )
                    else:
                        selected_expert = self.default_expert_name
                else:
                    selected_expert = task_name
                load_experts.append(self.experts[selected_expert])
            # assume all experts are loras
            output = LoRA.parallel_linear_forward(input, load_experts)
        else:
            ##########
            ## no experts, experts were merged into the layer
            output = self.layer(input, **kwargs)
        return output


# # same as smear, but uses merging after the ouyter product
# @register_modifier("multi_expert_routing")
# def modify_with_w_routing(transformer, config):
#     import re
#     from src.graph.module_graph import ModuleGraph
#     from mmlu_exper_merge_nevergrad import get_module_graph
#     module_graph, _ = get_module_graph(config.module_graph)
#     module_graph = re.sub(r"\$([a-zA-Z_][a-zA-Z0-9_]*)","1",module_graph)
#     graph = ModuleGraph.from_string(module_graph)


#     config.router_selector = "w_routing"
#     config.adapter_type = "lora"

#     if config.adapter_type in ["lora"]:
#         return modify_from_gaph(
#             transformer, config, graph, action="route", adapter_klass=AuxRoutingLoRALinear_MergeAfterOP
#         )
#     else:
#         raise NotImplementedError(
#             f"Adapter type {config.adapter_type} not implemented for vsmear modifier."
#         )
