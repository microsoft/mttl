from abc import abstractproperty
from pyparsing import abstractmethod
import torch
from torch import nn
from typing import Any, Dict
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


class Selector:
    @abstractmethod
    def forward(self, input, **kwargs) -> list:
        pass

    @abstractmethod
    def get_routing_weights(self):
        pass

    @abstractproperty
    def name(self):
        pass

    @abstractmethod
    def add_expert(self, expert_name: str):
        pass


@register_multi_expert_selector("poly_router")
class MultiExpertSelector(torch.nn.Module, Selector):
    """
    Implements routing at a per-layer or pe-model level
    """

    def __init__(self, config, info_container=None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.expert_names: list = []

        self.module_logits = nn.Parameter(torch.empty(1).uniform_(-1e-3, 1e-3))

        self.__layer_name__ = f"poly_router"

    def add_experts(self, expet_names: list):
        for expert_name in expet_names:
            self.add_expert(expert_name)

    @property
    def name(self):
        return f"{self.__layer_name__}"

    def forward(self, *args, **kwargs):
        module_logits = torch.sigmoid(self.module_logits)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)
        return [{k: v for k, v in zip(self.expert_names, module_weights)}]

    def get_routing_weights(self):
        return self.forward()[0]

    def add_expert(self, expert_name: str):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)
            self.module_logits.data = torch.empty(len(self.expert_names)).uniform_(
                -1e-3, 1e-3
            )


@register_multi_expert_selector("task_selector")
class TaskNameSelector(torch.nn.Module, Selector):
    def __init__(self, config=None, info_container=None, **kwargs) -> None:
        super().__init__()
        self.info_container = info_container
        self.__layer_name__ = f"task_selector"
        self.expert_names = []
        self.default_expert_name = None

    def forward(self, *args, **kwargs):
        task_names = self.info_container["routing_infos"].task_names

        if (
            any(task_name not in self.expert_names for task_name in task_names)
            and not self.default_expert_name
            and len(self.expert_names)
        ):
            raise ValueError(
                "Experts for all tasks have not been loaded! Set a default expert?"
            )

        routing_weights = [{task_name: 1.0} for task_name in task_names]

        return routing_weights

    def add_expert(self, expert_name: list):
        if expert_name not in self.expert_names:
            self.expert_names.append(expert_name)

    def add_experts(self, expet_names: list):
        for expert_name in expet_names:
            self.add_expert(expert_name)

    @property
    def name(self):
        return f"{self.__layer_name__}"
