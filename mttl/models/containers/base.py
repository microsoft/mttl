import abc
from typing import Dict, List, Union

import torch
from pyparsing import abstractmethod
from torch import Tensor, nn

from mttl.config import Config
from mttl.logging import warn_once
from mttl.models.containers.selectors import (
    BatchExpertsAndWeightsSelectorOutput,
    BatchExpertsSelectorOutput,
    BatchSequenceExpertsAndWeightsSelectorOutput,
    ExpertsAndWeightsSelectorOutput,
    KVTaskNameSelector,
    Selector,
    SelectorOutput,
)
from mttl.models.library.expert import Expert
from mttl.models.modifiers.base import ModifierConfig, ModifyMixin
from mttl.models.modifiers.kv_adapter import KVAdapter, KVAdapterConfig
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRA, SkilledLoRAConfig
from mttl.models.modifiers.modify_model import get_modifier_name


class Container(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class ExpertContainer(nn.Module, Container):
    __supports_configs__ = []

    def __init__(self, config, layer, selector=None):
        super().__init__()

        from mttl.models.containers.selectors import TaskNameSelector

        self.config = config
        self.layer = layer
        self.selector = selector or TaskNameSelector()
        self._default_expert_name = None
        self.expert_infos = {}

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

        update = action != "merge"
        if update:
            self.expert_infos[expert.name] = expert_info
            self.default_expert_name: str | None = (
                expert.name if is_default else self.default_expert_name
            )

        self.on_add_expert(expert, action=action, is_default=is_default)
        if update:
            # if a new expert was added, we update the selector and information meta-data
            self.selector.add_expert(
                expert.name, expert_info=expert_info, is_default=is_default
            )

    @property
    def expert_names(self) -> list:
        return list(self.expert_infos.keys())

    def _check_config(self, expert_config: Union[Config, ModifierConfig]):
        """Checks if the config is supported and converts it to the supported config type if needed."""
        if isinstance(expert_config, Config):
            # patches the config to be a LoRAConfig for the future
            from mttl.models.modifiers.base import ModifierConfig

            expert_config = ModifierConfig.from_training_config(expert_config)

        if type(expert_config) not in self.__supports_configs__:
            raise ValueError(
                "Unsupported expert config type {} for this type of expert container.".format(
                    type(expert_config)
                )
            )

    def export_experts(self) -> List[Expert]:
        experts = []
        for name, expert_module in self.experts.items():
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
    def forward(self, input, **kwargs):
        pass

    def get(self, key: Union[int, str]):
        if type(key) == int:
            key = self.expert_names[key]

        if key not in self.experts:
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

    def __len__(self):
        return len(self.expert_names)
