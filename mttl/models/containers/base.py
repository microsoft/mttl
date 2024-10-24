import abc
from typing import List, Union

import torch
from pyparsing import abstractmethod
from torch import nn

from mttl.models.containers.selectors.base import Selector, TaskNameSelector
from mttl.models.containers.utils import create_modif_regex, match_modules_to_modify
from mttl.models.library.expert import Expert
from mttl.models.modifiers.base import ModifierConfig
from mttl.utils import logger


class ContainerFullException(Exception):
    def __init__(self):
        super().__init__("Container is full. Cannot add more experts.")


class Container(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @classmethod
    def modify_transformer(
        cls,
        transformer,
        expert: Expert,
        action="route",
        is_default=False,
        selector_config=None,
        selector_cache=None,
    ):
        pass


class MergeableContainer(abc.ABC):
    @abstractmethod
    def merge_expert(self, expert_name):
        pass


class ExpertContainer(nn.Module, Container):
    __supports_configs__ = []

    def __init__(self, config, layer, selector=None):
        super().__init__()

        self._enabled = True
        self.config = config
        self.layer = layer
        self.selector = selector or TaskNameSelector()
        self._default_expert_name = None
        self.expert_infos = {}

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    def merge_with_layer(self):
        raise NotImplementedError("This container does not support merging.")

    def merge_expert(self, expert_name: str):
        raise NotImplementedError("This container does not support merging.")

    @property
    def default_expert_name(self):
        return self._default_expert_name

    @default_expert_name.setter
    def default_expert_name(self, value):
        self._default_expert_name = value
        if self.selector is not None:
            self.selector.default_expert_name = value

    def assign_selector(self, selector: Selector) -> None:
        """Assigns a selector to this container."""
        del self.selector
        self._modules.pop("selector", None)

        # propagate experts to the selector
        self.selector = selector
        # dependency injection on layer name
        self.selector.__layer_name__ = self.layer_name + ".selector"

        for expert_name, expert_info in self.expert_infos.items():
            self.selector.add_expert(
                expert_name,
                expert_info=expert_info,
                is_default=expert_name == self.default_expert_name,
            )

    def add_expert(self, expert: Expert, action="route", is_default=False) -> None:
        expert_info = expert.expert_info

        if expert.name in self.expert_infos:
            raise ValueError(
                "An expert with name {} already exists.".format(expert.name)
            )

        if is_default and action == "merge":
            raise ValueError(
                "Cannot set is_default if this expert is merged, change to 'route'."
            )

        self.expert_infos[expert.name] = expert_info
        self.default_expert_name: str | None = (
            expert.name if is_default else self.default_expert_name
        )

        self.on_add_expert(expert, is_default=is_default)

        if action != "merge":
            # if a new expert was added, we update the selector and information meta-data
            self.selector.add_expert(
                expert.name, expert_info=expert_info, is_default=is_default
            )
        else:
            self.merge_expert(expert.name)

    @property
    def expert_names(self) -> list:
        return list(self.expert_infos.keys())

    def _check_config(self, expert_config: ModifierConfig):
        """Checks if the config is supported and converts it to the supported config type if needed."""
        if not isinstance(expert_config, ModifierConfig):
            raise ValueError(
                "Expert config must be of type ModifierConfig, but got {}.".format(
                    type(expert_config)
                )
            )

        if type(expert_config) not in self.__supports_configs__:
            raise ValueError(
                "Unsupported expert config type {} for this type of expert container.".format(
                    type(expert_config)
                )
            )

    def export_experts(self) -> List[Expert]:
        experts = []
        for name in self.expert_names:
            expert_module = self.get(name)
            expert = Expert(
                expert_info=self.expert_infos[name],
                expert_weights={
                    self.layer_name + "." + n: v for n, v in expert_module.state_dict()
                },
            )
            experts.append(expert)
        return experts

    @abstractmethod
    def on_add_expert(
        self,
        expert: Expert,
        action="merge",
        is_default=False,
    ) -> None:
        pass

    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError("Dependency injection for layer name has not been done.")

        return self.__layer_name__

    @abstractmethod
    def container_forward(self, input, **kwargs):
        pass

    def forward(self, input, **kwargs):
        if not len(self) or not self._enabled:
            return self.layer(input, **kwargs)
        return self.container_forward(input, **kwargs)

    def get(self, key: Union[int, str]):
        if type(key) == int:
            key = self.expert_names[key]

        if key not in self.expert_infos:
            if self.default_expert_name is None:
                raise ValueError(
                    "Expert with name {} does not exist and no default expert is set.".format(
                        key
                    )
                )
            return self[self.default_expert_name]
        return self[key]

    def get_merged_params(self, with_global_names=True, **merger_kwargs):
        """
        Merges experts to one expert according to selector weights.
        """
        merged_params = {}
        merging_weights = self.selector.get_merging_weights(
            **merger_kwargs
        )  # expert_name: weight
        for exp_name, merging_weight in merging_weights.items():
            for k, parameter in self[exp_name].state_dict().items():
                key = k if not with_global_names else self.layer_name + "." + k
                if k not in merged_params:
                    merged_params[key] = parameter * merging_weight
                else:
                    merged_params[key] += parameter * merging_weight

        return merged_params

    @classmethod
    def modify_transformer(
        cls,
        transformer,
        expert: Expert,
        action: str = "route",
        is_default: bool = False,
        selector_config: "SelectorConfig" = None,
        selector_cache: "SelectorsCache" = None,
    ) -> None:
        """
        Base routine to modify the transformer architecture with an expert.

        Params:
            transformer: the transformer model to modify
            expert: expert instance that needs to be added
            action: whether to route or merge this expert, default is `route`
            is_default: whether the expert should be set as default
            selector_config: selector configuration to use for the model
            selector_cache: cache to store the selectors for the model
        """
        from mttl.models.modifiers.modify_model import get_modifier_name

        expert_config = expert.expert_config

        if not expert.name:
            raise ValueError("Expert name cannot be empty!")

        total_layers = 0
        added_layers = []
        added_containers = []
        model_modifier = get_modifier_name(expert_config)

        modify_modules = create_modif_regex(
            expert_config.modify_modules, expert_config.modify_layers
        )
        for m_name, module in match_modules_to_modify(transformer, modify_modules):
            # no layers to modify, try modifying the module
            total_layers += 1
            module_name = f"{m_name}"

            if not isinstance(module, ExpertContainer):
                expert_container = cls(
                    expert_config,
                    module,
                )
                expert_container.__layer_name__ = module_name

                parent_name, child_name = m_name.rsplit(".", 1)
                parent_module = dict(transformer.named_modules())[parent_name]
                setattr(
                    parent_module,
                    child_name,
                    expert_container,
                )
                added_containers.append(expert_container)
            else:
                expert_container = module

                if type(expert_container) != cls:
                    raise ValueError(
                        f"Module {m_name} was already modified by another container type."
                    )

            added_layers.append(expert_container.__layer_name__)
            expert_container.add_expert(
                expert,
                action=action,
                is_default=is_default,
            )

        if expert_config.tie_params:
            ### PARAM TYING ###
            # Note: because experts are added into expert containers
            # instead of parameter names being e.g. model.layers.4.self_attn.q_proj.lora_a,
            # it will be model.layers.4.self_attn.q_proj.experts.module1.lora_a

            # For this reason tying with q_proj\\.lora_a|k_proj\\.lora_a|v_proj\\.lora_a will not work,
            # and it has to be q_proj.*\\.lora_a|k_proj.*\\.lora_a|v_proj.*\\.lora_a
            from mttl.models.modifiers.base import (
                get_target_2_source_param_mapping,
                tie_params,
            )

            target_2_source_param = get_target_2_source_param_mapping(
                transformer.named_parameters(), expert_config.tie_params
            )
            tie_params(transformer, expert_config, target_2_source_param)

        if not added_layers:
            raise ValueError(
                "You were trying to add an expert but no expert containers were created, this is likely due to a misconfiguration of the expert config."
                " `modify_layers` and `modify_modules` did not return a match for the current model."
            )

        if selector_config is not None:
            from mttl.models.containers import replace_selector_for_container

            replace_selector_for_container(
                transformer,
                model_modifier,
                selector_config,
                selector_cache,
            )

            if not selector_cache.get(model_modifier):
                raise ValueError(
                    "No selectors were created but a routing config was specified. Check your selector_config and model architecture."
                )

            logger.debug(
                "Added expert %s, with %s selectors",
                expert.name,
                len(selector_cache.get(model_modifier)),
            )

        logger.debug("Patched layers: %s", added_layers)

    def __len__(self):
        return len(self.expert_names)


def containers_iterator(model: torch.nn.Module, return_parent=False):
    """Iterates over all containers in the model."""
    for mn, m in model.named_modules():
        for cn, c in m.named_children():
            if isinstance(c, ExpertContainer):
                yield (m, cn, c) if return_parent else c


def clear_containers(model: torch.nn.Module) -> torch.nn.Module:
    """Clears all containers in the model, just reassigning the layer to the model."""
    for m, cn, c in containers_iterator(model, return_parent=True):
        setattr(m, cn, c.layer)
    return model
