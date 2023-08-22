import torch
import copy
import torch.nn as nn   
from enum import Enum
from dataclasses import dataclass
import torch.nn.functional as F
import re 
import numpy as np       
from types import MethodType
from torch.autograd import Function
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from mttl.models.modify_model import patch_layers, register_modifier

from .utils import RoutingInfo

from mttl import global_vars
from mttl.global_vars import EPS

from projects.instr_routing.models.attention import SelectAttention    


SELECTORS = {}


def register_selector(name):
    def register_selector_cls(cls):
        if name in SELECTORS:
            raise ValueError(f"Cannot register duplicate selector ({name})")
        if not issubclass(cls, RoutingSelector):
            raise ValueError(
                f"Selectors ({name}: {cls.__name__}) must extend Selector"
            )
        SELECTORS[name] = cls
        return cls

    return register_selector_cls


def get_selector(config, **kwargs):
    if config.router_selector not in SELECTORS:
        raise ValueError(f"Cannot find selector: {config.router_selector}")
    return SELECTORS[config.router_selector](config, **kwargs)


class RouterWrapper:
    """Wrap transformer-based models with router-related functionalities.
    """
    @classmethod
    def register_functions(cls, object):
        methods = [
            method
            for method in dir(cls)
            if not method.startswith("__") and not "register_functions" in method
        ]

        for method in methods:
            print("Registering method: ", method)
            setattr(object, method, MethodType(getattr(cls, method), object))
        return object

    @classmethod
    def set_selector(
        cls,
        object,
        config,
        selector_to_replace,
        new_selector,
        **kwargs,
    ):
        """Switches PolytroponSelector to AverageSelector."""
        for name, module in object.named_modules():
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, selector_to_replace):
                    print(f"Replacing with {new_selector}: ", name)
                    setattr(
                        module,
                        name,
                        new_selector(config, **kwargs),
                    )

    @classmethod
    def switch_selector_to_average(cls, object, selector_to_replace, **kwargs):
        """Switches PolytroponSelector to AverageSelector."""
        for name, module in object.named_modules():
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, selector_to_replace):
                    print("Replacing with average: ", name)
                    setattr(
                        module,
                        name,
                        AverageSelector(**kwargs),
                    )

    @classmethod
    def get_routing_losses(cls, object):
        aux_losses = {}

        for name, adapter in object.get_adapters().items():
            if getattr(adapter, 'losses'):
                aux_losses[name] = adapter.losses
        return aux_losses

    @classmethod
    def clear_routing_losses(cls, object):
        for _, adapter in object.get_adapters().items():
            if getattr(adapter, 'losses'):
                adapter.losses = []

    @classmethod
    def clear_routing_metrics(cls, object):
        for _, adapter in object.get_adapters().items():
            if getattr(adapter, 'metrics'):
                adapter.metrics = {}

    @classmethod
    def get_routing_metrics(cls, object):
        metrics = {}

        for name, adapter in object.get_adapters().items():
            if getattr(adapter, 'metrics'):
                for n, v in adapter.metrics.items():
                    metrics[name + "." + n] = v
        return metrics

    @classmethod
    def get_adapters(cls, object):
        adapters = {}
        for n, m in object.named_modules():
            if isinstance(m, RoutingAdapter):
                adapters[n] = m
        return adapters

    @classmethod
    def get_selectors(cls, object):
        selectors = {}
        added_selectors = set()

        for name, adapter in object.get_adapters().items():
            # selectors might be shared across adapters
            if adapter.selector not in added_selectors:
                added_selectors.add(adapter.selector)
                selectors[name + ".selector"] = adapter.selector
        return selectors


class RoutingSelector(nn.Module):
    pass


class RoutingAdapter(nn.Module):
    @property
    def routing_infos(self) -> RoutingInfo:
        return self.task_id_ptr["routing_infos"]


@register_selector('average')
class AverageSelector(RoutingSelector):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        self.register_buffer(
            "module_logits", torch.empty(config.n_splits, config.n_skills).fill_(1.0 / config.n_skills)
        )

    def forward(self, routing_infos, **kwargs):
        bs = routing_infos.task_ids.size(0)
        module_logits = self.module_logits.view(1, self.n_splits, self.n_skills)
        return module_logits.expand(bs, -1, -1)


@register_selector('private')
class PrivateSelector(RoutingSelector):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.n_skills = config.n_skills

    def forward(self, routing_infos, **kwargs):
        return F.one_hot(routing_infos.task_ids, num_classes=self.n_skills).unsqueeze(1)
