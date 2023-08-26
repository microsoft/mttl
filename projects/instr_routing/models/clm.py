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
from mttl.models.modify_model import modify_transformer
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from mttl.models.get_scheduler import get_scheduler
from mttl.models.utils import (
    EfficientCheckpointModule,
    get_global_batch_size,
    RoutingInfo,
)
from mttl.models.routing import RouterWrapper
from mttl.models.get_optimizer import get_optimizer
from mttl.global_vars import EPS
from pytorch_lightning.utilities.parsing import AttributeDict
from dataclasses import dataclass


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
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

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

        # get some losses from the model if it is a router
        if type(self.model) == RouterWrapper:
            aux_loss = list(
                itertools.chain(*list(self.model.get_routing_losses().values()))
            )
            # we accumulate metrics over the microbatches
            if not self.training:
                # only plot this for validation
                for k, v in self.model.get_routing_metrics().items():
                    if "model.layers." in k:
                        self.accumulate_metrics_batch[k].append(v)

            self.model.clear_routing_losses()
            self.model.clear_routing_metrics()
        else:
            aux_loss = []
        return loss, aux_loss

    def calculate_routing_mask(self, inputs, labels=None):
        if not hasattr(self.hparams, "xrouting_option"):
            return None, None

        # bs, seq = x.shape
        padding_mask = (inputs != self.pad_token_id).float()
        # 1 if the token is part of instruction or pad token (so outputs are 0s)
        if labels is not None:
            instruction_mask = (labels == -100).float()
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
            inputs=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **kwargs
        )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # self.log_routing_metrics() .
        return super().on_before_optimizer_step(optimizer)

    def training_step(self, batch, _):
        loss, aux_loss = self.forward(batch)

        aux_loss_mean = torch.mean(torch.stack(aux_loss)) if len(aux_loss) > 0 else 0
        total_loss = loss + aux_loss_mean

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/aux_loss", aux_loss_mean, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_epoch=True, prog_bar=True)

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])

        return total_loss

    def log_routing_metrics(self, stage="train"):
        # we need to keep online mean ove rthe metrics over mictobatches (s.t. the metrics are calculated for the whole batch and not microbatches)
        divs, entropies, specializations, names = [], [], [], []

        for k, v in self.accumulate_metrics_batch.items():
            if (
                not "model.layers." in k
            ):  # we dont want to log the metrics for each layer
                # self.log(f"train/{k}", torch.tensor(v).mean(), on_step=True)
                pass
            else:
                names.append(k)
                # log per lyer metrics: -MI and entropy, calculated over minimatch
                layer_routing_dist = torch.cat(v, dim=0)
                layer_routing_dist = layer_routing_dist.view(
                    -1, layer_routing_dist.shape[-2], layer_routing_dist.shape[-1]
                )
                bs = layer_routing_dist.shape[0]
                n_skills, n_splits = self.hparams.n_skills, self.hparams.n_splits
                # calculate entropy and diversity over the full batch
                mixing_weights_ = layer_routing_dist.view(
                    -1, n_skills
                )  # .detach() # ex x n_skills
                mixing_weights_mean = layer_routing_dist.transpose(0, 1).mean(
                    dim=1
                )  # n_splits x n_skills

                average_normalized_entropy = (
                    -torch.sum(
                        mixing_weights_ * torch.log(mixing_weights_ + EPS), dim=-1
                    )
                    / np.log(n_skills)
                    if n_skills > 1
                    else torch.ones_like(mixing_weights_[:, 0])
                )  # ex
                # solit in n_splits chunks
                average_normalized_entropy = average_normalized_entropy.reshape(
                    bs, n_splits
                ).mean(
                    dim=0
                )  # bs
                # how different are the routinf for different examples? calculate MI, entropy of average - average entropy
                mixing_weights_mean = layer_routing_dist.transpose(0, 1).mean(
                    dim=1
                )  # n_splits x n_skills
                entropy_of_av_normalized = (
                    -torch.sum(
                        mixing_weights_mean * torch.log(mixing_weights_mean + EPS),
                        dim=-1,
                    )
                    / np.log(n_skills)
                    if n_skills > 1
                    else torch.zeros_like(mixing_weights_mean[0])
                )  # ex
                div = (
                    entropy_of_av_normalized - average_normalized_entropy
                ).mean()  # mean over n_splits
                entropy = average_normalized_entropy.mean()  # .item()
                specialization = div - entropy
                # if not len(self.loggers)>0 or not isinstance(self.loggers[0], pl.loggers.wandb.WandbLogger):
                # self.log(f"{stage}/{k}_div", div)#, on_step=True)
                # self.log(f"{stage}/{k}_entropy", entropy)#, on_step=True)
                divs.append(div.float().item())
                entropies.append(entropy.float().item())
                specializations.append(specialization.float().item())

        # log mean over all layers divs and entropies
        if len(divs) > 0:
            self.log(
                f"{stage}/div_layers_mean", torch.tensor(divs).mean()
            )  # , on_step=True)
            self.log(
                f"{stage}/entropy_layers_mean", torch.tensor(entropies).mean()
            )  # , on_step=True)
            self.log(
                f"{stage}/diversity(H-MI)_layers_mean",
                torch.tensor(specializations).mean(),
            )  # , on_step=True)
            # log distribution
            if (
                len(self.loggers) > 0
                and isinstance(self.loggers[0], pl.loggers.wandb.WandbLogger)
                and stage == "val"
            ):
                wandb_logger = self.loggers[0]
                # bar plot with reduced memory size
                plt.clf()
                _ = plt.plot(range(len(divs)), divs)
                wandb_logger.log_image(
                    f"{stage}/div_layers_dist",
                    [wandb.Image(plt)],
                    step=self.global_step,
                )  # , commit=False)
                # wandb_logger.log_table(f"{stage}/div_layers_dist_table", columns=names, data=torch.stack(divs).unsqueeze(0).tolist())
                # reset plot
                plt.clf()
                _ = plt.plot(range(len(entropies)), entropies)
                wandb_logger.log_image(
                    f"{stage}/entropy_layers_dist",
                    [wandb.Image(plt)],
                    step=self.global_step,
                )  # , commit=False)
                # wandb_logger.log_table(f"{stage}/entropy_layers_dist_table", columns=names, data=torch.stack(entropies).unsqueeze(0).tolist())
                plt.clf()
                _ = plt.plot(range(len(specializations)), specializations)
                wandb_logger.log_image(
                    f"{stage}/diversity(MI-H)_layers_dist",
                    [wandb.Image(plt)],
                    step=self.global_step,
                )  # , commit=False)
                # wandb_logger.log_table(f"{stage}/diversity(H-MI)_layers_dist_table", columns=names, data=torch.stack(specializations).unsqueeze(0).tolist())
                plt.clf()

                # create csv table if not exists
                csv_filename = (
                    f"{self.hparams.output_dir}/{stage}/div_layers_dist_table.csv"
                )
                if not os.path.exists(csv_filename):
                    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                    writer = csv.writer(open(csv_filename, "a"))
                    writer.writerow(names)
                with open(csv_filename, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(divs)

                csv_filename = (
                    f"{self.hparams.output_dir}/{stage}/entropy_layers_dist_table.csv"
                )
                if not os.path.exists(csv_filename):
                    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                    writer = csv.writer(open(csv_filename, "a"))
                    writer.writerow(names)
                with open(csv_filename, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(entropies)

                csv_filename = f"{self.hparams.output_dir}/{stage}/diversity(MI-H)_layers_dist_table.csv"
                if not os.path.exists(csv_filename):
                    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                    writer = csv.writer(open(csv_filename, "a"))
                    writer.writerow(names)
                with open(csv_filename, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(specializations)

        self.accumulate_metrics_batch = defaultdict(list)

    def on_validation_epoch_start(self) -> None:
        self.accumulate_metrics_batch = defaultdict(list)
        return super().on_validation_epoch_start()

    def log_aux_loss_per_layer(self, aux_loss):
        if isinstance(self.loggers[0], pl.loggers.wandb.WandbLogger):
            wandb_logger = self.loggers[0]
            plt.clf()
            aux_loss = [l.detach().item() for l in aux_loss]
            _ = plt.plot(range(len(aux_loss)), aux_loss)
            wandb_logger.log_image(
                "val/aux_loss_per_layer",
                [wandb.Image(plt)],
                step=self.global_step,
            )  # , commit=False)
            plt.clf()

    def log_xrouter_W_norm(self):
        if isinstance(self.loggers[0], pl.loggers.wandb.WandbLogger):
            from .routing import XRouter

            norms = []
            for n, layer in self.model.named_modules():
                if isinstance(layer, XRouter):
                    norms.append(layer.W_norm)
            if len(norms) > 0:
                wandb_logger = self.loggers[0]
                plt.clf()
                _ = plt.plot(range(len(norms)), norms)
                wandb_logger.log_image(
                    "val/xrouter_W_norm",
                    [wandb.Image(plt)],
                    step=self.global_step,
                )
                plt.clf()

    def validation_step(self, batch, batch_idx, log=True):
        loss, aux_loss = self.forward(batch, reduction="none")
        total_loss = loss

        aux_loss_mean = torch.mean(torch.stack(aux_loss)) if len(aux_loss) > 0 else 0

        mean_loss = total_loss.sum() / loss.size(0)

        if log:
            self.log("val/loss", mean_loss, on_epoch=True, prog_bar=True)
            self.log("val/aux_loss", aux_loss_mean, on_epoch=True, prog_bar=True)
            self.log_aux_loss_per_layer(aux_loss)
            if (
                batch_idx
                % (self.hparams.gradient_accumulation_steps * self.hparams.micro_batch_size)
                == 0
                and batch_idx > 0
            ):  # to accumulate over larger batch
                self.log_routing_metrics(stage="val")

        self._inference_outputs += [(loss, batch["task_ids"])]

        return loss, batch["task_ids"]

    def on_before_backward(self, loss: Tensor) -> None:
        return super().on_before_backward(loss)

    def test_step(self, batch, batch_idx):
        loss, aux_loss = self.forward(batch, reduction="none")
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
        self.accumulate_metrics_batch = defaultdict(list)
        self.log_routing_metrics(stage="val")
        self.log_xrouter_W_norm()
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
