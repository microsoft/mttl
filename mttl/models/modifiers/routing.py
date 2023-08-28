import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from dataclasses import dataclass
from typing import List
import re

from mttl.models.adapters import Adapter


SELECTORS = {}


def register_selector(name):
    def register_selector_cls(cls):
        if name in SELECTORS:
            raise ValueError(f"Cannot register duplicate selector ({name})")
        if not issubclass(cls, RoutingSelector):
            raise ValueError(f"Selectors ({name}: {cls.__name__}) must extend Selector")
        SELECTORS[name] = cls
        return cls

    return register_selector_cls


def get_selector(config, **kwargs):
    if config.router_selector not in SELECTORS:
        raise ValueError(f"Cannot find selector: {config.router_selector}")
    return SELECTORS[config.router_selector](config, **kwargs)


class RouterWrapper:
    """Wrap transformer-based models with router-related functionalities."""

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
            if getattr(adapter, "losses", None):
                aux_losses[name] = adapter.losses
        return aux_losses

    @classmethod
    def clear_routing_losses(cls, object):
        for _, adapter in object.get_adapters().items():
            if getattr(adapter, "losses", None):
                adapter.losses = []

    @classmethod
    def clear_routing_metrics(cls, object):
        for _, adapter in object.get_adapters().items():
            if getattr(adapter, "metrics", None):
                adapter.metrics = {}

    @classmethod
    def get_routing_metrics(cls, object):
        metrics = {}

        for name, adapter in object.get_adapters().items():
            if getattr(adapter, "metrics", None):
                for n, v in adapter.metrics.items():
                    metrics[name + "." + n] = v
        return metrics

    @classmethod
    def get_adapters(cls, object):
        adapters = {}
        for n, m in object.named_modules():
            if isinstance(m, Adapter):
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


@register_selector("average")
class AverageSelector(RoutingSelector):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.n_splits = config.n_splits
        self.n_skills = config.n_skills
        self.register_buffer(
            "module_logits",
            torch.empty(config.n_splits, config.n_skills).fill_(1.0 / config.n_skills),
        )

    def forward(self, routing_infos, **kwargs):
        bs = routing_infos.task_ids.size(0)
        module_logits = self.module_logits.view(1, self.n_splits, self.n_skills)
        return module_logits.expand(bs, -1, -1)


@register_selector("private")
class PrivateSelector(RoutingSelector):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.n_skills = config.n_skills

    def forward(self, routing_infos, **kwargs):
        return F.one_hot(routing_infos.task_ids, num_classes=self.n_skills).unsqueeze(1)


@dataclass
class RoutingInfo:
    task_ids: torch.Tensor = None
    hashes: List[str] = None
    instruction_hashes: List[str] = None
    example_ids: List[int] = None
    labels: torch.Tensor = None

    @classmethod
    def from_batch(cls, batch: dict):
        ri = cls(
            task_ids=batch.get("task_ids", None),
            hashes=batch.get("hashes", None),
            example_ids=batch.get("example_ids", None),
            instruction_hashes=batch.get("instruction_hashes", None),
            labels=batch.get("labels", None),
        )
        return ri

    def repeat_interleave(self, repeats):
        # useful for beam search
        self.task_ids = self.task_ids.repeat_interleave(repeats)
        if self.hashes:
            self.hashes = [h for h in self.hashes for _ in range(repeats)]
        if self.instruction_hashes:
            self.instruction_hashes = [h for h in self.instruction_hashes for _ in range(repeats)]
        self.example_ids = (
            self.example_ids.repeat_interleave(repeats)
            if self.example_ids is not None
            else None
        )


class RoutingMixin:
    def __init__(self, task_id_ptr) -> None:
        self.task_id_ptr = task_id_ptr

    @property
    def routing_infos(self) -> RoutingInfo:
        return self.task_id_ptr["routing_infos"]


def modify_with_routing(transformer, config, layer_type, optional_wrapper=None):
    # How to "bin" different levels of selectors ?
    def _extract_identifier(string, match_on="coder"):
        """Returns a unique identifier for the "chunk" of layers sharing the
        same underlying selector
        # e.g. 'block' : 'encoder.block.0.layer.0.SelfAttention' -> 'encoder.block.0'
        """
        pattern_map = {
            "coarsegrained": None,
            "finegrained": None,
            "layerwise": "layer",
            "blockwise": "block",
            "coderwise": "coder",
        }
        assert match_on in pattern_map.keys()

        if match_on == "finegrained":
            return string
        if match_on == "coarsegrained":
            return ""

        match_on = pattern_map[match_on]
        left_idx = string.find(f"{match_on}.") + len(match_on) + 1
        right_idx = string[left_idx:].find(".")
        return string[: left_idx + right_idx]

    selectors = {}
    total_layers = 0

    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.modify_layers, c_name):
                    identifier = _extract_identifier(
                        f"{m_name}.{c_name}", config.router_granularity
                    )
                    if identifier not in selectors.keys():
                        selectors[identifier] = get_selector(
                            config,
                            in_d=layer.in_features,
                        )

                    selector = selectors[identifier]
                    total_layers += 1

                    print(f"Patching {m_name}.{c_name}...")

                    wrapper = layer_type(
                        config,
                        transformer.task_id_container,
                        layer,
                        selector=selector,
                    )
                    wrapper.__layer_name__ = f"{m_name}.{c_name}"

                    setattr(
                        module,
                        c_name,
                        wrapper,
                    )

    print(
        f"created {len(selectors)} selectors for a total of {total_layers} adapted layers"
    )

    if optional_wrapper is not None:
        return optional_wrapper.register_functions(transformer)
    return transformer
