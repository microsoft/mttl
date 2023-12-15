from abc import ABC, abstractmethod
from torch import nn
import re
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

    @classmethod
    def from_training_config(cls, training_config: "Config"):
        """Build modifier config from the training config."""
        kwargs = {}
        for key, value in cls.__dict__.items():
            if key in training_config.__dict__:
                kwargs[key] = training_config.__dict__[key]
        return cls(**kwargs)


class AutoModifierConfig(object):
    @staticmethod
    def from_training_config(training_config: "Config"):
        """Build modifier config from the training config."""
        from mttl.models.modifiers.modify_model import MODIFIERS_TO_CONFIGS

        if training_config.model_modifier is None:
            raise ValueError("Model modifier not set in the training config!")

        if training_config.model_modifier not in MODIFIERS_TO_CONFIGS:
            raise ValueError(
                f"Model modifier '{training_config.model_modifier}' not found, has it been registered?"
            )

        config_klass = MODIFIERS_TO_CONFIGS[training_config.model_modifier]
        return config_klass.from_training_config(training_config)


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
