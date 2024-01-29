import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from dataclasses import dataclass, field, fields
from typing import Dict, List
import re

from mttl.models.modifiers.base import Adapter, ModifyMixin
from mttl.utils import logger


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
        success = False
        for name, module in object.named_modules():
            for name, inner_mod in module.named_children():
                if isinstance(inner_mod, selector_to_replace):
                    print("Replacing with average: ", name)
                    setattr(
                        module,
                        name,
                        AverageSelector(**kwargs),
                    )
                    success = True
        return success

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
    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError(
                "Layer name not available, dependency injection not done properly?"
            )

        return self.__layer_name__

    def __init__(self, config):
        super().__init__()

        self.config = config


@register_selector("average")
class AverageSelector(RoutingSelector):
    def __init__(self, config, **kwargs):
        super().__init__(config)

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
        super().__init__(config)

    def forward(self, routing_infos, **kwargs):
        return routing_infos.task_ids


@dataclass
class RoutingInfo:
    task_ids: torch.Tensor = None
    task_names: List[str] = None
    example_ids: List[int] = None
    labels: torch.Tensor = None
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    task_weights: torch.nn.ParameterDict = None
    aux_losses: Dict = field(default_factory=dict)

    @classmethod
    def from_batch(cls, batch: dict, **kwargs):
        task_ids = batch.get("task_ids").long() if "task_ids" in batch else None
        task_names = batch.get("task_names", None)
        task_weights = batch.get("task_weights", None)

        ri = cls(
            task_ids=task_ids,
            task_names=task_names,
            task_weights=task_weights,
            input_ids=batch.get("input_ids", None),
            example_ids=batch.get("example_ids", None),
            labels=batch.get("labels", None),
            attention_mask=batch.get("attention_mask", None),
            **kwargs,
        )
        return ri

    def _repeat(self, inputs, n):
        if inputs is not None:
            if isinstance(inputs, torch.Tensor):
                return inputs.repeat_interleave(n)
            else:
                return [item for item in inputs for _ in range(n)]
        return inputs

    def repeat_interleave(self, repeats):
        # useful for beam search
        self.task_ids = self._repeat(self.task_ids, repeats)
        self.task_names = self._repeat(self.task_names, repeats)
        self.example_ids = self._repeat(self.example_ids, repeats)
        self.task_weights = self._repeat(self.task_weights, repeats)


class RoutingMixin:
    def __init__(self, info_container, *args, **kwargs) -> None:
        self._info_container = info_container

    @property
    def info_container(self):
        return self._info_container

    @property
    def routing_infos(self) -> RoutingInfo:
        return self.info_container.get("routing_infos", None)


class RouterModifyMixin(ModifyMixin):
    @classmethod
    def modify_transformer(cls, transformer, config, optional_wrapper=None):
        return modify_with_routing(cls, transformer, config, RouterWrapper)


def modify_with_routing(cls, transformer, config, optional_wrapper=None):
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
                    layer_name = f"{m_name}.{c_name}"

                    identifier = _extract_identifier(
                        f"{m_name}.{c_name}", config.router_granularity
                    )

                    if identifier not in selectors.keys():
                        # Special case when you have a decoder layer in an enc-dec model
                        if (
                            not ("encoder" in m_name)
                            and config.model_family == "encdec"
                        ):
                            from transformers.models.t5.modeling_t5 import (
                                T5ForConditionalGeneration,
                            )

                            assert isinstance(transformer, T5ForConditionalGeneration)
                            in_d = transformer.config.d_model
                        else:
                            # The following line will fail if `layer` is not a `nn.Linear` layer
                            in_d = getattr(layer, "in_features", None)

                        selectors[identifier] = get_selector(
                            config,
                            in_d=in_d,
                        )
                        selectors[identifier].__layer_name__ = layer_name + ".selector"

                    selector = selectors[identifier]
                    total_layers += 1

                    logger.info(f"Patching {m_name}.{c_name}...")

                    wrapper = cls(
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
