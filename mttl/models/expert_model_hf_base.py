import dataclasses
import inspect
import json
import math
import os
import re
import threading
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Union

import torch
from attrs import field
from huggingface_hub import hf_hub_download
from torch.optim.optimizer import Optimizer

from mttl.config import Args, ExpertConfig, MoEExpertConfig, MultiExpertConfig
from mttl.logging import logger
from mttl.models.containers import add_expert_to_transformer
from mttl.models.containers.base import ExpertContainer
from mttl.models.containers.selectors.base import (
    LoadableLibraryMixin,
    LoadableSelectorConfig,
    MultiSelectorConfig,
    Selector,
    SelectorConfig,
    SelectorsCache,
)
from mttl.models.expert_context import InfoContainer
from mttl.models.expert_model_hf_config import AutoModelConfig, BaseExpertModelConfig
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.models.modifiers.lora import SkilledLoRAConfig
from mttl.models.modifiers.modify_model import get_modifier_name
from mttl.models.utils import (
    EfficientCheckpointModule,
    model_loader_helper,
    prepare_model_for_kbit_training,
)
from mttl.registrable import Registrable

torch.set_float32_matmul_precision("high")


WEIGHTS_NAME = "mttl_weights.bin"
LOADING_KWARGS_NAME = "loading_kwargs.json"


class BaseExpertModel(torch.nn.Module, Registrable):
    def __init__(
        self, config: BaseExpertModelConfig, model_object=None, **loading_kwargs
    ):
        super().__init__()

        # log hyperparameters
        self.load_in_4bit = loading_kwargs.get("load_in_4bit", None) or False
        self.load_in_8bit = loading_kwargs.get("load_in_8bit", None) or False

        self.model = (
            model_loader_helper(
                config.base_model,
                load_in_4bit=self.load_in_4bit,
                load_in_8bit=self.load_in_8bit,
                device_map=loading_kwargs.get("device_map", "cpu"),
                attn_implementation=loading_kwargs.get("attn_implementation", None),
            )
            if model_object is None
            else model_object
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
        **model_kwargs: Any,
    ):
        # get model config first
        model_config = AutoModelConfig.from_pretrained(model_id)

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

        model = model_class(model_config, **model_kwargs)
        load_status = model.load_state_dict(
            torch.load(weights_file, weights_only=True), strict=False
        )
        if len(load_status.unexpected_keys):
            raise ValueError("Unexpected keys found in the state dict.")
        return model

    @InfoContainer.wrap_forward
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        task_names=None,
        task_ids=None,
        task_sources=None,
        reduction="mean",
        **kwargs,
    ):
        outputs = self.model.forward(input_ids, attention_mask)

        if labels is not None:
            # calculate loss, could also be done inside of the model
            bs = input_ids.size(0)
            logits = outputs.logits
            vocab_size = logits.size(-1)
            labels = labels.squeeze(-1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            # reshape back
            if reduction == "none":
                loss = loss.view((bs, -1))
                # mean only non-zero
                non_zero_loss = (loss != 0).sum(dim=-1)
                non_zero_loss[non_zero_loss == 0] = 1
                loss = loss.sum(dim=-1) / non_zero_loss
            return loss, outputs
        return outputs

    @property
    def generation_config(self):
        return self.model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.model.generation_config = value

    @InfoContainer.wrap_forward
    def generate(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        task_names=None,
        task_ids=None,
        task_sources=None,
        **kwargs,
    ):
        generations = self.model.generate(
            inputs=input_ids, attention_mask=attention_mask, **kwargs
        )
        return generations
