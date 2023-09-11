import json
import os
import csv
import torch
import copy

import pytorch_lightning as pl
from typing import Any, List, Dict
from collections import defaultdict
from torch import nn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.routing import RoutingInfo, RoutingSelector
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from mttl.models.get_scheduler import get_scheduler
from mttl.models.utils import (
    EfficientCheckpointModule,
    get_global_batch_size,
)
from mttl.models.modifiers.experts import add_expert_to_transformer
from mttl.utils import get_checkpoint_path, logger
from mttl.models.get_optimizer import get_optimizer
from config import ExpertConfig
from dataclasses import dataclass, field


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


class MultiExpertModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters(ignore=["tokenizer", "model_object"])

        self.tokenizer = kwargs["tokenizer"]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model: AutoModelForCausalLM = None

        if "llama" in self.hparams.model:
            model_object = LlamaForCausalLM.from_pretrained(
                self.hparams.model,
                load_in_8bit=self.hparams.load_in_8bit,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        else:
            model_object = AutoModelForCausalLM.from_pretrained(self.hparams.model)

        if model_object.config.vocab_size != len(self.tokenizer):
            model_object.resize_token_embeddings(len(self.tokenizer))

        if self.hparams.load_in_8bit:
            model_object = prepare_model_for_kbit_training(model_object)

        self.model = model_object

    def load_expert(
        self, expert_path: str, expert_name: str = None, action: str = "merge"
    ):
        # load the expert weights
        import json
        import os

        if expert_name is None:
            expert_name = os.path.basename(expert_path)

        expert_checkpoint = get_checkpoint_path(expert_path)

        logger.info(f"Loading expert from {expert_checkpoint}...")

        expert_checkpoint = torch.load(expert_checkpoint, map_location="cpu")
        expert_config = ExpertConfig(
            kwargs=expert_checkpoint["hyper_parameters"],
            silent=True,
            raise_error=False
        )

        expert_weights = expert_checkpoint["state_dict"]
        expert_weights = {k.replace("model.", "", 1): v for k, v in expert_weights.items()}

        if self.hparams.model != expert_config.model:
            raise ValueError(
                "The expert has been trained on top of a different model!"
                " Detected: {} - Expected: {}".format(
                    expert_config.model, self.hparams.model
                )
            )

        logger.info(
            f"Adding expert with name {expert_name}... with action ... {action}!"
        )

        self.model = add_expert_to_transformer(
            self.model, expert_name, expert_config, expert_weights, action=action
        )

    @property
    def generation_config(self):
        return self.model.generation_config

    def generate(
        self,
        batch,
        routings=None,
        save_oracle_routings=None,
        **kwargs,
    ):
        if hasattr(self.model, "task_id_container"):
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )

        generations = self.model.generate(inputs=batch["input_ids"], **kwargs)
        return generations
