from abc import ABC, abstractmethod
from typing import Dict, Union
from torch import nn
import re
from mttl.models.modifiers.modify_model import (
    CONFIGS_TO_MODIFIERS,
    MODIFIERS_TO_CONFIGS,
)
from mttl.utils import logger
from dataclasses import dataclass


class Adapter(nn.Module):
    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError(
                "Layer name not set, dependency injection not done properly?"
            )

        return self.__layer_name__


class MergeableAdapter(ABC, Adapter):
    @abstractmethod
    def merge_with_layer(self):
        pass


@dataclass
class ModifierConfig(object):
    modify_modules: str = ".*"
    modify_layers: str = ".*"

    def __eq__(self, other):
        # compare all the attributes
        return self.__dict__ == other.__dict__

    def asdict(self) -> Dict:
        """Dump the config to a string."""
        from dataclasses import asdict

        data = asdict(self)
        # store the model modifier for easy loading
        data["modifier_config_klass"] = self.__class__.__name__
        return data

    @classmethod
    def from_training_config(cls, training_config: "Config"):
        """Build modifier config from the training config."""
        kwargs = {}
        for key, _ in cls.__dataclass_fields__.items():
            if hasattr(training_config, key):
                kwargs[key] = getattr(training_config, key)
        return cls(**kwargs)

    @classmethod
    def fromdict(cls, dumped: Dict) -> "ModifierConfig":
        klass = dumped.pop("modifier_config_klass")
        return eval(klass)(**dumped)

    @staticmethod
    def from_training_config(
        training_config: Union["Config", "ModifierConfig"]
    ) -> Union["ModifierConfig", None]:
        """Build modifier config from the training config.

        Returns None if no modifier is set.
        """
        from mttl.models.modifiers.modify_model import MODIFIERS_TO_CONFIGS

        if isinstance(training_config, ModifierConfig):
            # nothing to do here
            return training_config

        if training_config.model_modifier is None:
            return None

        if training_config.model_modifier not in MODIFIERS_TO_CONFIGS:
            raise ValueError(
                f"Model modifier '{training_config.model_modifier}' not found, has it been registered?"
            )

        config_klass = MODIFIERS_TO_CONFIGS[training_config.model_modifier]
        kwargs = {}
        for key, _ in config_klass.__dataclass_fields__.items():
            if hasattr(training_config, key):
                kwargs[key] = getattr(training_config, key)
        return config_klass(**kwargs)


class ModifyMixin(nn.Module):
    @classmethod
    def modify_transformer(cls, transformer, config):
        return modify_with_adapter(transformer, config, cls)


def modify_with_adapter(transformer, config, adapter_klass):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.modify_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.modify_layers, c_name):
                    logger.info(f"Patching {m_name}.{c_name}...")

                    setattr(
                        module,
                        c_name,
                        adapter_klass(config, layer),
                    )
    return transformer
