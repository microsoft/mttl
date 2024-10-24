import inspect
import json
import os
from typing import Any, Dict, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from transformers.modeling_outputs import CausalLMOutput

from mttl.logging import logger
from mttl.models.expert_config import AutoModelConfig, BaseExpertModelConfig
from mttl.models.expert_context import InfoContainer
from mttl.models.utils import model_loader_helper
from mttl.registrable import Registrable

torch.set_float32_matmul_precision("high")


WEIGHTS_NAME = "mttl_weights.bin"
LOADING_KWARGS_NAME = "loading_kwargs.json"


def filter_kwargs(func, kwargs):
    return {k: v for k, v in kwargs.items() if k in inspect.signature(func).parameters}


class BaseExpertModel(torch.nn.Module, Registrable):
    def __init__(
        self, config: BaseExpertModelConfig, model_object=None, **loading_kwargs
    ):
        super().__init__()

        # log hyperparameters
        self.load_in_4bit = loading_kwargs.get("load_in_4bit", False)
        self.load_in_8bit = loading_kwargs.get("load_in_8bit", False)
        self.device_map = loading_kwargs.get("device_map", "cpu")
        self.precision = loading_kwargs.get("precision", "bf16")
        self.attn_implementation = loading_kwargs.get("attn_implementation", None)

        # cannot use both load_in_4bit and load_in_8bit
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both `load_in_4bit` and `load_in_8bit`.")

        if config.base_model is None and model_object is None:
            raise ValueError("You must provide a model object or a base model name.")

        # if model_object is provided, base_model should be None
        if model_object is not None and config.base_model is not None:
            logger.warning(
                "You are initializing a model directly from a model object. We will set `base_model` to None."
            )
            config.base_model = None

        self.model = (
            model_loader_helper(
                config.base_model,
                bf16=self.precision == "bf16",
                fp16=self.precision == "fp16",
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                device_map=self.device_map,
                attn_implementation=self.attn_implementation,
            )
            if model_object is None
            else model_object
        )

        if model_object:
            logger.warning(
                "You are initializing a model directly from a model object. The same object module will need to be used for re-loading."
            )

        self.config = config
        self.loading_kwargs = loading_kwargs

    def _delete_non_trainable_params(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Delete all parameters that are not marked as trainable."""
        if not hasattr(self, "_params_from_checkpoint"):
            self._params_from_checkpoint = set()

        if not hasattr(self, "trainable_param_names"):
            self.trainable_param_names = [
                n for n, p in self.named_parameters() if p.requires_grad
            ]

        keys = [k for k in state_dict.keys()]

        deleted = []
        for key in keys:
            # we can safely avoid dumping this parameter if it is both
            # not in the trainable parameters and was not loaded from checkpoint
            if not (key in self.trainable_param_names) and not (
                key in self._params_from_checkpoint
            ):
                del state_dict[key]
                deleted.append(key)

        logger.info("Deleted from state dict: {}".format(len(deleted)))
        return state_dict

    # write a repr function
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    def save_pretrained(self, save_directory, **kwargs):
        """Bare bone save pretrained function that saves the model and config."""
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        os.makedirs(save_directory, exist_ok=True)
        state_dict = self._delete_non_trainable_params(self.state_dict())
        output_config_file = self.config.save_pretrained(save_directory, **kwargs)
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(state_dict, output_model_file)
        torch.save(
            json.dumps(self.loading_kwargs),
            os.path.join(save_directory, LOADING_KWARGS_NAME),
        )
        return output_model_file, output_config_file

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, os.PathLike],
        model_object: Optional[torch.nn.Module] = None,
        **model_kwargs: Any,
    ):
        # get model config first
        model_config: BaseExpertModelConfig = AutoModelConfig.from_pretrained(model_id)

        if os.path.isfile(os.path.join(model_id, WEIGHTS_NAME)):
            weights_file = os.path.join(model_id, WEIGHTS_NAME)
        else:
            try:
                weights_file = hf_hub_download(model_id, WEIGHTS_NAME)
            except Exception as exc:
                raise ValueError(f"Can't find {WEIGHTS_NAME} at '{model_id}'") from exc

        if cls == BaseExpertModel:
            model_class = BaseExpertModel.get_class_by_config_class(model_config)
        else:
            model_class = cls

        if model_config.base_model is None and model_object is None:
            raise ValueError(
                "Base model name was None in the checkpoint. You must provide a model object."
            )

        model = model_class(model_config, model_object=model_object, **model_kwargs)
        load_status = model.load_state_dict(
            torch.load(weights_file, weights_only=True), strict=False
        )
        if len(load_status.unexpected_keys):
            raise ValueError("Unexpected keys found in the state dict.")
        return model

    @InfoContainer.create_context
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ) -> CausalLMOutput:
        outputs = self.model.forward(
            input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )
        return outputs

    @property
    def generation_config(self):
        return self.model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.model.generation_config = value

    @InfoContainer.create_context
    def generate(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        generations = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return generations
