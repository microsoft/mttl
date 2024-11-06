import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Union

from torch import nn

from mttl.logging import logger
from mttl.registrable import Registrable
from mttl.serializable import AutoSerializable, Serializable
from mttl.utils import deprecated


class Modifier(nn.Module, Registrable):
    # default modifier
    default = "lora"

    def __init__(self):
        super().__init__()

    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError(
                "Layer name not set, dependency injection not done properly?"
            )

        return self.__layer_name__

    @classmethod
    def modify_transformer(cls, transformer, config):
        return modify_with_adapter(transformer, config, cls)


class MergeableModifierMixin(ABC):
    @abstractmethod
    def merge_with_layer(self):
        pass


@dataclass
class ModifierConfig(Serializable):
    modify_modules: str = ".*"
    modify_layers: str = (
        None  # this is depriciated but still kept for backward compatibility
    )
    tie_params: str = None

    @property
    def modifier_name(self):
        return Modifier.get_name_by_config_class(type(self))

    @classmethod
    def from_training_config(
        cls, training_config: Union["Args", "ModifierConfig"]
    ) -> Union["ModifierConfig", None]:
        """Build modifier config from the training config.

        Returns None if no modifier is set.
        """
        from mttl.arguments import create_config_class_from_args

        if isinstance(training_config, ModifierConfig):
            # nothing to do here
            return training_config

        if cls.__name__ == "ModifierConfig":
            if training_config.model_modifier is None:
                return None

            if training_config.model_modifier not in Modifier.registered_names():
                raise ValueError(
                    f"Model modifier '{training_config.model_modifier}' not found, has it been registered?"
                )

            config_klass = Modifier.get_config_class_by_name(
                training_config.model_modifier
            )
        else:
            config_klass = cls
        return create_config_class_from_args(config_klass, training_config)


class AutoModifierConfig(AutoSerializable):
    @classmethod
    @deprecated(
        message="The config appears to be a legacy config and will be discontinued in the next release."
    )
    def fromdict_legacy(cls, data) -> "ModifierConfig":
        modifier_name = data.pop("__model_modifier__")
        config_cls: ModifierConfig = Modifier.get_config_class_by_name(modifier_name)
        return config_cls.fromdict(data)

    @classmethod
    def fromdict(cls, data: Dict) -> "ModifierConfig":
        try:
            return AutoSerializable.fromdict(data)
        except ValueError:
            return cls.fromdict_legacy(data)


def get_target_2_source_param_mapping(
    named_params: Iterable, tie_params, expand_if_targets_are_missing=False
) -> Dict[str, str]:
    """
    Create a dict for parameter tying: target param -> source param
    Assumes:
        - that "tie_params" is a regex that matches the parameters that need to be tied.
        - matched parameters will be tied within a common parent module.

    Constraints:
        - tie_params should match the parameter and not the module, e.g. it should be "q_proj.lora_a|k_proj.lora_a" and not "q_proj|k_proj".

    Example for Llama like model:
        - if tie_params = "q_proj.*\\.lora_a|k_proj.*\\.lora_a|v_proj.*\\.lora_a", lora_a will be tied across q_proj, k_proj, and v_proj within the same parent (attn_module).
        - if tie_params = ".*lora_a.*", nothing will be tied because lora_a is not within the same parent module.

    Args:
        param_names: Iterable of tuples (param_name, any)
        tie_params: regex that matches the parameters that need to be tied within the same parent.
        expand_if_targets_are_missing: bool, if True, will add missing targets to target_2_source_param at first match without checking if targets are also in named_params. Should be used when name_params only contains the sources.

    """
    valid_tie_names = [p for p, _ in named_params]
    target_2_source_param = {}  # target param -> source param
    if tie_params:
        source_2_parameter = {}
        for p_name in valid_tie_names:
            match = re.search(tie_params, p_name)
            if match:
                matched_option = match.group()
                parent = p_name.split(matched_option)[0]
                assert (
                    f"{parent}{matched_option}" in valid_tie_names
                ), "tie_params should match the parameter and not the module, e.g. it should be 'q_proj.lora_a|k_proj.lora_a' and not 'q_proj|k_proj', revise your tie_params argument."
                if parent in source_2_parameter:
                    if not expand_if_targets_are_missing:
                        target_2_source_param[p_name] = source_2_parameter[parent]
                    assert p_name in target_2_source_param
                else:
                    source_2_parameter[parent] = p_name

                    if expand_if_targets_are_missing:
                        # we end up here because we are adding missing targets to target_2_source_param (probably called from expert.py)
                        # we need to remove all regex specific syntax from the tie_params
                        params_to_add = (
                            tie_params.replace("\\", "").replace(".*", "").split("|")
                        )
                        for p in params_to_add:
                            if p not in p_name:
                                target_2_source_param[f"{parent}{p}"] = p_name

    return target_2_source_param


def get_parameter_object(transformer, p_name_query):
    """
    Given a model and a parameter name, returns the parameter object.
    """
    for m_name, module in dict(transformer.named_modules()).items():
        for p_name, param in dict(module.named_parameters(recurse=False)).items():
            if f"{m_name}.{p_name}" == p_name_query:
                return getattr(module, p_name)


def tie_params(transformer, config, target_2_source_param):
    """
    Given a dict for parameter tying: target param -> source param, ties the parameters.
    """
    if len(target_2_source_param) > 0:
        for m_name, module in dict(transformer.named_modules()).items():
            for p_name, param in dict(module.named_parameters(recurse=False)).items():
                if f"{m_name}.{p_name}" in target_2_source_param:
                    logger.info(
                        f"Tying {m_name}.{p_name} to {target_2_source_param[f'{m_name}.{p_name}']}..."
                    )
                    # m_name is the common parent module,
                    # but to keep it more general we retrieve the module by parameter name again
                    p_source = get_parameter_object(
                        transformer, target_2_source_param[f"{m_name}.{p_name}"]
                    )

                    setattr(module, p_name, p_source)
                    assert getattr(module, p_name) is p_source
        assert len(transformer.state_dict().keys()) > len(
            list(transformer.named_parameters())
        ), "Some parameters are not tied."


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

    target_2_source_param = get_target_2_source_param_mapping(
        transformer.named_parameters(), config.tie_params
    )

    tie_params(transformer, config, target_2_source_param)
    return transformer
