from ast import Dict
import json
import os
import csv
import torch
import copy
import itertools
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
import wandb
from typing import List
from collections import defaultdict
from torch import Tensor, nn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.routing import RoutingInfo, RoutingSelector
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from mttl.models.get_scheduler import get_scheduler
from mttl.models.utils import (
    EfficientCheckpointModule,
    get_global_batch_size,
)
from mttl.models.get_optimizer import get_optimizer
from mttl.global_vars import EPS
from dataclasses import dataclass


EPS = 1e-12
        
def entropy(mixing_weights):
    return (
        -torch.sum(
            mixing_weights * torch.log(mixing_weights + EPS), dim=-1
        )
    )
    
@dataclass
class AugmentedRoutingInfo(RoutingInfo):
    save_oracle_routings: bool = False
    gen_mode: bool = False
    routings: List[torch.Tensor] = None
    oracle_routings: List[torch.Tensor] = None
    pad_token_mask: torch.Tensor = None
    inst_token_mask: torch.Tensor = None


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


class CLM(EfficientCheckpointModule):
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
                    torch_dtype=torch.float32,
                    device_map="auto",
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

    @property
    def generation_config(self):
        return self.model.generation_config

    def add_loss_plugin(self, plugin):
        if self.loss_plugins is not None:
            self.loss_plugins[plugin.name] = plugin
        else:
            self.loss_plugins = nn.ModuleDict({plugin.name: plugin})

    def forward(self, batch, reduction="mean"):
        input_ids, labels = batch["input_ids"], batch["labels"]
        routing_infos = AugmentedRoutingInfo.from_batch(batch)

        pad_mask, instruction_mask = self.calculate_routing_mask(
            batch["input_ids"], batch["labels"]
        )
        routing_infos.pad_token_mask = pad_mask
        routing_infos.inst_token_mask = instruction_mask

        self.model.task_id_container["routing_infos"] = routing_infos
        assert (
            routing_infos.pad_token_mask.shape[1]
            == routing_infos.inst_token_mask.shape[1]
        )

        outputs = self.model.forward(input_ids, attention_mask=pad_mask)

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

    def calculate_routing_mask(self, inputs, labels=None):
        # 1 if the token is not a pad token (so inputs and outputs are 1)
        padding_mask = (inputs != self.pad_token_id).float()
        if labels is not None:
            # 1 if the token is part of instruction (so outputs and pad tokens are 0s)
            instruction_mask = (labels == -100).float() * padding_mask
        else:
            instruction_mask = padding_mask.clone()
        return padding_mask, instruction_mask

    def compute_routings(self, batch, **kwargs):
        out = self.generate(batch, save_oracle_routings=True, gen_mode=0, **kwargs)
        oracle_routings = self.model.task_id_container["routing_infos"].oracle_routings
        return out, oracle_routings

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.model.zero_grad() 
        return super().on_before_zero_grad(optimizer)

    def generate(
        self,
        batch,
        **kwargs,
    ):
        if not hasattr(self.model, "task_id_container"):
            self.model.task_id_container = {}

        routing_infos = AugmentedRoutingInfo.from_batch(batch)
        routing_infos.gen_mode = 1

        pad_mask, instruction_mask = self.calculate_routing_mask(batch["input_ids"])
        routing_infos.pad_token_mask = pad_mask
        routing_infos.inst_token_mask = instruction_mask

        # if routings are given (should be oracle routings), we will use them for generation
        if "routings" in kwargs:
            routing_infos.routings = kwargs["routings"]
            kwargs.pop("routings")

        # if flag is set, we will store the oracle routings
        if "save_oracle_routings" in kwargs:
            routing_infos.save_oracle_routings = kwargs["save_oracle_routings"]
            kwargs.pop("save_oracle_routings")

        if "gen_mode" in kwargs:  # so that in xr4 we look at both nput and output
            routing_infos.gen_mode = kwargs["gen_mode"]

        self.model.task_id_container["routing_infos"] = routing_infos

        return self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # self.log_routing_metrics() .
        return super().on_before_optimizer_step(optimizer)

    def gather_auxiliary_losses(self):
        # get some losses from the model if it is a router
        aux_loss = []
        for name, module in self.model.named_modules():
            if isinstance(module, RoutingSelector) and hasattr(module, "auxiliary_loss"):
                aux_loss_mod = getattr(module, 'auxiliary_loss', None)
                if aux_loss_mod is not None:
                    aux_loss.append(aux_loss_mod)
        return aux_loss

    def training_step(self, batch, _):
        loss = self.forward(batch) 

        aux_loss = self.gather_auxiliary_losses()
        aux_loss = torch.stack(aux_loss).sum() if len(aux_loss) else 0.
        total_loss = loss + aux_loss

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/aux_loss", aux_loss, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_epoch=True, prog_bar=True)

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss

    def validation_step(self, batch, batch_idx, log=True):
        loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        aux_loss = self.gather_auxiliary_losses()
        aux_loss_sum = torch.stack(aux_loss).sum() if len(aux_loss) else 0.

        if log:
            self.log("val/loss", mean_loss, on_epoch=True, prog_bar=True)
            self.log("val/aux_loss", aux_loss_sum, on_epoch=True, prog_bar=True)

        self._inference_outputs += [(loss, batch["task_ids"])]
        return loss, batch["task_ids"]

    def on_before_backward(self, loss: Tensor) -> None:
        return super().on_before_backward(loss)

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        self._inference_outputs += [(loss, batch["task_ids"])]
        return loss, batch["task_ids"]

    def on_test_epoch_end(self):
        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)
        task_ids = torch.cat([out[1] for out in outputs], 0)
        log_name = f"test/loss"

        if hasattr(self.model, "checkpoint_tested"):
            log_name = f"test/{self.model.checkpoint_tested}/loss"

        # log per task loss and overall loss
        self.log(log_name, losses.mean(), on_epoch=True, prog_bar=True)

        for task_id in torch.unique(task_ids):
            log_name = f"test/loss_{task_id.item()}"
            if hasattr(self.model, "checkpoint_tested"):
                log_name = f"test/{self.model.checkpoint_tested}/loss_{task_id.item()}"
            self.log(
                log_name,
                losses[task_ids == task_id].mean(),
                on_epoch=True,
                prog_bar=True,
            )

        self.accumulate_metrics_batch = defaultdict(list)
        self._inference_outputs.clear()
        return losses

    def on_validation_epoch_end(self):
        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)
        task_ids = torch.cat([out[1] for out in outputs], 0)

        # compute the loss per task id
        with open(
            os.path.join(self.hparams.output_dir, "val_loss_by_task.txt"), "a+"
        ) as f:
            task_losses = {}
            for task_id in torch.unique(task_ids):
                task_losses[task_id.item()] = losses[task_ids == task_id].mean().item()
            f.write(json.dumps(task_losses) + "\n")
        self._inference_outputs.clear()

    def configure_optimizers(self):
        args = self.hparams
        self.ml_optimizer = self.ml_scheduler = None

        optimizer, self.trainable_param_names = get_optimizer(
            self, args, no_decay=["bias", "LayerNorm.weight"]
        )
        global_bs = get_global_batch_size(
            args.train_batch_size, args.gradient_accumulation_steps
        )

        if args.total_steps == -1:
            args.total_steps = (
                len(self.trainer.datamodule.train_dataset) // global_bs
            ) * self.trainer.max_epochs

        if args.warmup_steps == -1:
            args.warmup_steps = int(args.warmup_proportion * args.total_steps)

        # args.scheduler = "linear_decay_with_warmup"
        scheduler = get_scheduler(optimizer, args)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @property
    def hparams_initial(
        self,
    ):  # to make wandb logger work we need to override this method
        """The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.
        Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.

        Returns:
            AttributeDict: immutable initial hyperparameters
        """
        if not hasattr(self, "_hparams_initial"):
            return AttributeDict()
        # prevent any change
        hparams_initial = copy.deepcopy(self._hparams_initial)
        # pop anything that is not json serializable
        hparams_initial.pop("_updated_kwargs")
        return hparams_initial
