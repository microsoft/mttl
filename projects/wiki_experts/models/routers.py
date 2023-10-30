import math
import torch
import torch.nn as nn
from enum import Enum

from mttl.models.adapters import Adapter, MergableAdapter
from typing import Any, Dict
import torch.nn.functional as F
from mttl.models.modifiers.experts import add_expert_to_transformer
from mttl.models.adapters import SkilledLoRA, LoRA, SkilledLoRA_MergeLoraAfterOP
from abc import abstractmethod, ABCMeta, abstractproperty

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

    @abstractmethod
    def get_routing_weights(self):
        pass

    @abstractproperty
    def name(self):
        pass


class LocalRouter:
    @abstractmethod
    def forward(self, input, **kwargs):
        pass

    @abstractmethod
    def get_routing_weights(self):
        pass

    @abstractproperty
    def name(self):
        pass


@register_multi_expert_selector("local_softmax")
class Local_Multi_ExpertRouter(torch.nn.Module, LocalRouter):
    """
    Implements routing at a per-layer level: the routing weights shared accross all layers of the model
    """

    # TODO: implement more complex versions of this router: one that ha smore params, that takes also the inputs ad routes, etc.
    def __init__(self, config, expert_names, routed_layer_name=None):
        super().__init__()
        self.config = config
        self.activtion = torch.nn.Softmax(dim=-1)
        self.expert_names = expert_names
        self.routed_layer_name = routed_layer_name
        self._merging_weights = torch.nn.parameter.Parameter(
            torch.ones(len(expert_names)), requires_grad=True
        )
        # init _merging_weights to be gaussian
        self._merging_weights.data.normal_(mean=0.0, std=0.02)
        self._is_initialized = True

    @property
    def name(self):
        return f"local_softmax_{self.routed_layer_name}"

    def forward(self, input, **kwargs):
        weights = self.activtion(self._merging_weights)
        return {k: v for k, v in zip(self.expert_names, weights)}

    def get_routing_weights(self):
        weights = self.activtion(self._merging_weights)
        return {k: v.cpu().item() for k, v in zip(self.expert_names, weights)}


@register_multi_expert_selector("global_softmax")
class Global_Multi_ExpertRouter(torch.nn.Module, GlobalRouter):
    """
    Implements routing at the level of the full model: the routing weights shared accross all layers
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        name = "global_softmax"
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

    def get_routing_weights(self):
        weights = self.activtion(self._merging_weights)
        return {k: v.cpu().item() for k, v in zip(self.expert_names, weights)}


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
