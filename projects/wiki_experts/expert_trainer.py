import json
import os
import csv
import torch
import copy
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
from mttl.models.get_optimizer import get_optimizer
from mttl.utils import get_mlf_logger, setup_logging, logger
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

        # FIX for enabling gradient of the auxiliary loss
        # enable gradient checkpointing for memory efficiency
        from functools import partial

        notfailing_checkpoint = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
        torch.utils.checkpoint.checkpoint = notfailing_checkpoint
        model.gradient_checkpointing_enable()

    return model


class ExpertTrainer(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log hyperparameters
        self.save_hyperparameters(ignore=["tokenizer", "model_object"])

        self.tokenizer = kwargs["tokenizer"]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model: AutoModelForCausalLM = None
        self.accumulate_metrics_batch = defaultdict(list)

        if kwargs.get("model_object") is None:
            if "llama" in self.hparams.model:
                model_object = LlamaForCausalLM.from_pretrained(
                    self.hparams.model,
                    load_in_8bit=self.hparams.load_in_8bit,
                    torch_dtype=torch.bfloat16,
                    device_map=getattr(self.hparams, "device_map", "auto"),
                )
            else:
                model_object = AutoModelForCausalLM.from_pretrained(self.hparams.model)

            if model_object.config.vocab_size != len(self.tokenizer):
                model_object.resize_token_embeddings(len(self.tokenizer))

            if self.hparams.load_in_8bit:
                model_object = prepare_model_for_kbit_training(model_object)

            self.model = modify_transformer(model_object, self.hparams)
        else:
            self.model = kwargs.get("model_object")

        self.loss_plugins = nn.ModuleDict({})
        self.test_results = []
        self.best_val_result = None
        self._inference_outputs = []

    def forward(self, batch, reduction="mean"):
        input_ids, labels = batch["input_ids"], batch["labels"]

        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)

        outputs = self.model.forward(input_ids, attention_mask=batch["attention_mask"])

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

        del outputs, shift_logits, shift_labels
        return loss

    def on_save_checkpoint(self, ckpt):
        if not hasattr(self, "_params_from_checkpoint"):
            self._params_from_checkpoint = set()

        # remove also parameters in the loss plugins, these need not be saved
        # (auxiliary parameters for the losses)
        plugin_param_keys = set()
        for _, plugin in self.loss_plugins.items():
            plugin_param_keys.update(plugin.state_dict().keys())

        keys = [k for k in ckpt["state_dict"].keys()]

        for key in keys:
            # we can safely avoid dumping this parameter if it is both
            # not in the trainable parameters and was not loaded from checkpoint
            if (
                not (key in self.trainable_param_names)
                and not (key in self._params_from_checkpoint)
            ) or key in plugin_param_keys:
                del ckpt["state_dict"][key]
                print("Deleting from state dict:", key)

    def training_step(self, batch, _):
        loss = self.forward(batch)
        total_loss = loss

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            "train/total_loss", total_loss, on_step=True, prog_bar=True
        )
        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]

        self._inference_outputs += [(loss.detach().cpu(),)]
        return loss

    def on_validation_epoch_end(self):
        from itertools import chain

        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)

        self._inference_outputs.clear()
        self.log("val/loss", losses.mean(), on_epoch=True, prog_bar=True)
