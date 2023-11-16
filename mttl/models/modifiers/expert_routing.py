from abc import abstractproperty
import re
from pyparsing import abstractmethod
import torch
from torch import nn
from typing import Any, Dict
from mttl.models.modifiers.base import MergeableAdapter
from mttl.models.modifiers.lora import LoRA, SkilledLoRA
from mttl.utils import logger


MULTI_EXPERT_ROUTERS = {}
EPS = 1e-8


def register_multi_expert_selector(name):
    print("Registering multi-expert selector..." + name)

    def _thunk(fn):
        if name in MULTI_EXPERT_ROUTERS:
            raise ValueError(
                f"Cannot register duplicate multi-expert selector ({name})"
            )
        MULTI_EXPERT_ROUTERS[name] = fn
        return fn

    return _thunk


def get_selector(config, info_container, **kwargs):
    if config.router_selector:
        if config.router_selector not in MULTI_EXPERT_ROUTERS:
            raise ValueError(f"Cannot find selector: {config.router_selector}")
        return MULTI_EXPERT_ROUTERS[config.router_selector](
            config, info_container, **kwargs
        )
    else:
        return None


class Router:
    @abstractmethod
    def forward(self, input, **kwargs):
        pass

    @abstractmethod
    def get_routing_weights(self):
        pass

    @abstractproperty
    def name(self):
        pass


def _extract_identifier(string, match_on="coder"):
    """Returns a unique identifier for the "chunk" of layers sharing the
    same underlying selector
    # e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
    """

    if match_on == "finegrained":
        return string
    if match_on == "coarsegrained":
        return " "
    return string


@register_multi_expert_selector("poly_router")
class Multi_ExpertRouter(torch.nn.Module, Router):
    """
    Implements routing at a per-layer or pe-model level
    """

    def __init__(self, config, info_container, expert_names=[]):
        super().__init__()
        self.config = config
        self.expert_names: list = expert_names
        self.info_container = info_container

        self.module_logits = nn.Parameter(
            torch.empty(len(expert_names)).uniform_(-1e-3, 1e-3)
        )

        self.__layer_name__ = f"poly_router"

    def resize_module_logits(self, expet_names: list):
        self.expert_names += expet_names
        self.module_logits.data = torch.empty(len(self.expert_names)).uniform_(
            -1e-3, 1e-3
        )

    @property
    def name(self):
        return f"{self.__layer_name__}"

    def forward(self, *args, **kwargs):
        module_logits = torch.sigmoid(self.module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return {k: v for k, v in zip(self.expert_names, module_weights)}

    def get_routing_weights(self):
        return self.forward()


@register_multi_expert_selector("kv_router")
class DummyKVRouter(Router, nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        assert False

    def resize_module_logits(self, *args, **kwargs):
        pass
