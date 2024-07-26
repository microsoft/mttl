import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Union

from torch import nn
from transformers import PreTrainedModel

from mttl.logging import logger
from mttl.registrable import Registrable


@dataclass
class ModifierConfig(object):
    modify_modules: str = ".*"
    modify_layers: str = ".*"
    tie_params: str = None

    def __eq__(self, other):
        # compare all the attributes
        return self.__dict__ == other.__dict__

    def asdict(self) -> Dict:
        """Dump the config to a string."""
        from dataclasses import asdict

        from mttl.models.modifiers.modify_model import CONFIGS_TO_MODIFIERS

        data = asdict(self)
        # store the model modifier for easy loading
        data["__model_modifier__"] = CONFIGS_TO_MODIFIERS[type(self)]
        return data

    @classmethod
    def fromdict(cls, dumped: Dict) -> "ModifierConfig":
        from mttl.models.modifiers.modify_model import MODIFIERS_TO_CONFIGS

        if "__model_modifier__" not in dumped:
            raise ValueError(
                "Cannot load config from dict, missing '__model_modifier__' key."
            )
        mod = dumped.pop("__model_modifier__")
        return MODIFIERS_TO_CONFIGS[mod](**dumped)

    @classmethod
    def from_training_config(
        cls, training_config: Union["Config", "ModifierConfig"]
    ) -> Union["ModifierConfig", None]:
        """Build modifier config from the training config.

        Returns None if no modifier is set.
        """
        from mttl.models.modifiers.modify_model import MODIFIERS_TO_CONFIGS

        if isinstance(training_config, ModifierConfig):
            # nothing to do here
            return training_config

        if cls != ModifierConfig:
            model_modifier = Modifier.get_name_by_config_class(cls)
        else:
            if training_config.model_modifier is None:
                return None

            model_modifier = training_config.model_modifier

        if model_modifier not in Modifier.registered_names():
            raise ValueError(
                f"Model modifier '{model_modifier}' not found, has it been registered?"
            )

        config_klass = Modifier.get_config_class_by_name(model_modifier)
        kwargs = {}
        for key, _ in config_klass.__dataclass_fields__.items():
            if hasattr(training_config, key):
                kwargs[key] = getattr(training_config, key)
        return config_klass(**kwargs)


class Modifier(nn.Module, Registrable):
    @property
    def layer_name(self):
        if not hasattr(self, "__layer_name__"):
            raise ValueError(
                "Layer name not set, dependency injection not done properly?"
            )

        return self.__layer_name__

    @classmethod
    def modify_transformer(
        cls, transformer: PreTrainedModel, modifier_config: ModifierConfig
    ) -> PreTrainedModel:
        """Applies the modifier to the transformer."""
        for m_name, module in dict(transformer.named_modules()).items():
            if re.fullmatch(modifier_config.modify_modules, m_name):
                for c_name, layer in dict(module.named_children()).items():
                    if re.fullmatch(modifier_config.modify_layers, c_name):
                        logger.info(f"Patching {m_name}.{c_name}...")

                        setattr(
                            module,
                            c_name,
                            cls(modifier_config, layer),
                        )

        # handle parameter tying if necessary
        target_2_source_param = get_target_2_source_param_mapping(
            transformer.named_parameters(), modifier_config.tie_params
        )

        tie_params(transformer, modifier_config, target_2_source_param)
        return transformer


class MergeableModifier(ABC, Modifier):
    @abstractmethod
    def merge_with_layer(self):
        pass


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
