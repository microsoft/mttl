import json
import os
import sys
import csv
import torch
import copy
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
import wandb
from collections import defaultdict
from torch import Tensor, nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

from .get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler
from mttl.models.modify_model import modify_transformer
from mttl.models.utils import EfficientCheckpointModule, get_global_batch_size
from .utils import RoutingInfo
from mttl.utils import freeze_embeds
from pytorch_lightning.utilities.parsing import AttributeDict

EPS = 1e-12


class CLM(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log hyperparameters
        self.save_hyperparameters(ignore=["tokenizer", "model_object"])
        self.args = self.hparams

        self.tokenizer = kwargs["tokenizer"]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model: AutoModelForCausalLM = None
        self.accumulate_metrics_batch = defaultdict(list)
        if kwargs.get("model_object") is None:
            raise NotImplementedError()
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.args.model,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", "/tmp/hf-cache"),
            )
            # free-up temporary space
            os.system("/bin/rm -rf /tmp/hf-cache")
            os.system("df")

            if "t5" or "T0" in self.args.model:
                self.pad_token_id = self.tokenizer.pad_token_id
                if (
                    hasattr(self.model.config, "max_position_embeddings")
                    and self.model.config.max_position_embeddings
                    < self.args.max_input_length
                ):
                    print(
                        f"Increasing the model's number of position embedding vectors from {self.model.config.max_position_embeddings} "
                        f"to {self.args.max_input_length}."
                    )
                    self.model.resize_position_embeddings(self.args.max_input_length)
            elif "bart" in self.args.model:
                self.pad_token_id = self.model.model.shared.padding_idx
            else:
                raise NotImplementedError()

            self.model = modify_transformer(self.model, self.args)
            print(self.args)

            if self.args.freeze_embeds:
                print("Freezing embeddings")
                freeze_embeds(self.model)
        else:
            self.model = kwargs.get("model_object")
        self.loss_plugins = nn.ModuleDict({})

        self.test_results = []
        self.best_val_result = None
        self._inference_outputs = []

    def add_loss_plugin(self, plugin):
        if self.loss_plugins is not None:
            self.loss_plugins[plugin.name] = plugin
        else:
            self.loss_plugins = nn.ModuleDict({plugin.name: plugin})

    def forward(self, batch, reduction="mean"):
        input_ids, labels = batch["input_ids"], batch["labels"]
        # print("input_ids", input_ids.shape)
        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
            batch
        )  # pad tokens also have -100
        padding_mask, instr_mask = self.calculate_routing_mask(
            batch["input_ids"], self.model.task_id_container["routing_infos"]
        )
        setattr(
            self.model.task_id_container["routing_infos"],
            "pad_token_mask",
            padding_mask,
        )  # if 1 token is either instruction or output, i.e. not a pad token
        setattr(
            self.model.task_id_container["routing_infos"], "inst_token_mask", instr_mask
        )  # 1 if the token is part of instruction or pad token (so outputs are 0s)

        outputs = self.model.forward(
            input_ids,
            attention_mask=(input_ids != self.pad_token_id).float(),
        )
        # output_ids = outputs.logits.argmax(-1)
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
        aux_loss = self.model.task_id_container["routing_infos"].aux_loss
        
        # aux_loss = (
        #     torch.mean(
        #         torch.stack(self.model.task_id_container["routing_infos"].aux_loss)
        #     )
        #     if len(self.model.task_id_container["routing_infos"].aux_loss) > 0
        #     else 0
        # )

        # we accumulate metrics over the microbatches
        if not self.training:  # only plot this for validation
            for k, v in self.model.task_id_container["routing_infos"].metrics.items():
                if "model.layers." in k:
                    self.accumulate_metrics_batch[k].append(v)

        return loss, aux_loss
                   
    def calculate_routing_mask(self, x, routing_infos):
        if not hasattr(self.args, "xrouting_option"):
            return None, None
        # bs, seq = x.shape
        # if self.args.xrouting_option > 0:
        padding_mask = (             
            routing_infos.pad_token_mask
        )  # 1 if the token is not a pad token, so its either instruciton or output
        instruction_mask = torch.ones_like(
            padding_mask
        )  # 1 if the token is part of instruction or pad token (so outputs are 0s)

        if routing_infos.labels is not None:
            instruction_mask = (
                routing_infos.labels == -100
            ).float()  # 1 if the token is part of instruction or pad token (so outputs are 0s)
        return padding_mask, instruction_mask

    def compute_routings(self, batch, **kwargs):
        out = self.generate(batch, save_oracle_routings=True, gen_mode=0, **kwargs)
        oracle_routings = self.model.task_id_container["routing_infos"].oracle_routings
        return out, oracle_routings

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.model.zero_grad()
        return super().on_before_zero_grad(optimizer)

    def generate(self, batch, **kwargs):
        if not hasattr(self.model, "task_id_container"):
            self.model.task_id_container = {}
        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)
        setattr(self.model.task_id_container["routing_infos"], "gen_mode", 1)
        padding_mask, instr_mask = self.calculate_routing_mask(
            batch["input_ids"], self.model.task_id_container["routing_infos"]
        )
        setattr(
            self.model.task_id_container["routing_infos"],
            "pad_token_mask",
            padding_mask,
        )  # if 1 token is either instruction or output, i.e. not a pad token
        setattr(
            self.model.task_id_container["routing_infos"], "inst_token_mask", instr_mask
        )  # 1 if the token is part of instruction or pad token (so outputs are 0s)

        # if routings are given (should be oracle routings), we will use them for generation
        if "routings" in kwargs:
            setattr(
                self.model.task_id_container["routing_infos"],
                "routings",
                kwargs["routings"],
            )
            kwargs.pop("routings")

        # if flag is set, we will store the oracle routings
        if "save_oracle_routings" in kwargs:
            setattr(
                self.model.task_id_container["routing_infos"],
                "save_oracle_routings",
                kwargs["save_oracle_routings"],
            )
            kwargs.pop("save_oracle_routings")

        if "gen_mode" in kwargs:  # so that in xr4 we look at both nput and output
            setattr(
                self.model.task_id_container["routing_infos"],
                "gen_mode",
                kwargs["gen_mode"],
            )
            kwargs.pop("gen_mode")

        return self.model.generate(inputs=batch["input_ids"], **kwargs)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # self.log_routing_metrics() .
        return super().on_before_optimizer_step(optimizer)

    def training_step(self, batch, _):
        loss, aux_loss = self.forward(batch)
        
        aux_loss_mean = (torch.mean(torch.stack(aux_loss))if len(aux_loss) > 0 else 0
        )
        
        total_loss = loss + aux_loss_mean
        # outputs = self.model.forward(**batch)
        # loss= outputs.loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/aux_loss", aux_loss_mean, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_epoch=True, prog_bar=True)

        for plugin in self.loss_plugins.values():
            plugin_loss = plugin.compute_loss(self.model, batch)
            loss += plugin.factor * plugin_loss
            self.log(
                f"train/{plugin.name}_loss", plugin_loss, on_epoch=True, prog_bar=True
            )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss

    def log_routing_metrics(self, stage="train"):
        # we need to keep online mean ove rthe metrics over mictobatches (s.t. the metrics are calculated for the whole batch and not microbatches)
        divs, entropies, specializations, names = [], [], [], []
        for k, v in self.accumulate_metrics_batch.items():
            # log to wandb directly
            # wandb.log({f"train/{k}": torch.tensor(v).mean()})
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
                n_skills, n_splits = self.args.n_skills, self.args.n_splits
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
                    f"{self.args.output_dir}/{stage}/div_layers_dist_table.csv"
                )
                if not os.path.exists(csv_filename):
                    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                    writer = csv.writer(open(csv_filename, "a"))
                    writer.writerow(names)
                with open(csv_filename, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(divs)

                csv_filename = (
                    f"{self.args.output_dir}/{stage}/entropy_layers_dist_table.csv"
                )
                if not os.path.exists(csv_filename):
                    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                    writer = csv.writer(open(csv_filename, "a"))
                    writer.writerow(names)
                with open(csv_filename, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(entropies)

                csv_filename = f"{self.args.output_dir}/{stage}/diversity(MI-H)_layers_dist_table.csv"
                if not os.path.exists(csv_filename):
                    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
                    writer = csv.writer(open(csv_filename, "a"))
                    writer.writerow(names)
                with open(csv_filename, mode="a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(specializations)

                # if not hasattr(self, f"{stage}_div_layers_dist_table"):
                #     setattr(self, f"{stage}_div_layers_dist_table", wandb.Table(columns=names))
                # getattr(self, f"{stage}_div_layers_dist_table").add_data(*divs)
                # if not hasattr(self, f"{stage}_entropy_layers_dist_table"):
                #     setattr(self, f"{stage}_entropy_layers_dist_table", wandb.Table(columns=names))
                # getattr(self, f"{stage}_entropy_layers_dist_table").add_data(*entropies)
                # if not hasattr(self, f"{stage}_diversity(MI-H)_layers_dist_table"):
                #     setattr(self, f"{stage}_diversity(MI-H)_layers_dist_table", wandb.Table(columns=names))
                # getattr(self, f"{stage}_diversity(MI-H)_layers_dist_table").add_data(*specializations)

                # wandb_logger.log_table(f"{stage}/div_layers_dist_table", columns=getattr(self, f"{stage}_div_layers_dist_table").columns, data = getattr(self, f"{stage}_div_layers_dist_table").data)
                # wandb_logger.log_table(f"{stage}/entropy_layers_dist_table", columns=getattr(self, f"{stage}_entropy_layers_dist_table").columns, data = getattr(self, f"{stage}_entropy_layers_dist_table").data)
                # wandb_logger.log_table(f"{stage}/diversity(MI-H)_layers_dist_table", columns=getattr(self, f"{stage}_diversity(MI-H)_layers_dist_table").columns, data = getattr(self, f"{stage}_diversity(MI-H)_layers_dist_table").data)

            #     _ = plt.bar(range(len(divs)), divs)
            #     self.loggers[0].log_image(f"{stage}/div_layers_dist", [wandb.Image(plt)], step=self.global_step, commit=False)
            #     _ = plt.bar(range(len(entropies)), entropies)
            #     self.loggers[0].log_image(f"{stage}/entropy_layers_dist", [wandb.Image(plt)], step=self.global_step, commit=False)

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
            wandb_logger.log_image("val/aux_loss_per_layer",
                        [wandb.Image(plt)],
                        step=self.global_step,
                    )  # , commit=False)
            plt.clf()         
     
    def log_xrouter_W_norm(self):
        if isinstance(self.loggers[0], pl.loggers.wandb.WandbLogger):
            from .routing import XRouter
            norms = []
            for n,layer in self.model.named_modules():
                if isinstance(layer, XRouter):
                    norms.append(layer.W_norm)
            if len(norms) > 0:
                    wandb_logger = self.loggers[0]
                    plt.clf()
                    _ = plt.plot(range(len(norms)), norms)   
                    wandb_logger.log_image("val/xrouter_W_norm",
                                [wandb.Image(plt)],
                                step=self.global_step,
                            )
                    plt.clf()
            
    
    def validation_step(self, batch, batch_idx, log=True):
        loss, aux_loss = self.forward(batch, reduction="none")
        total_loss = loss  # +aux_loss
        # outputs = self.model.forward(**batch, reduction='none')
        # loss= outputs.loss
        
                  
        aux_loss_mean = (torch.mean(torch.stack(aux_loss))if len(aux_loss) > 0 else 0)
        
        
        mean_loss = total_loss.sum() / loss.size(0)
        if log:
            self.log("val/loss", mean_loss, on_epoch=True, prog_bar=True)    
            self.log("val/aux_loss", aux_loss_mean, on_epoch=True, prog_bar=True)
            self.log_aux_loss_per_layer(aux_loss)
            if (batch_idx % (self.args.gradient_accumulation_steps * self.args.micro_batch_size)== 0 and batch_idx > 0):  # to accumulate over larger batch
                self.log_routing_metrics(stage="val")

        self._inference_outputs += [(loss, batch["task_ids"])]

        return loss, batch["task_ids"]

    def on_before_backward(self, loss: Tensor) -> None:
        return super().on_before_backward(loss)

    def test_step(self, batch, batch_idx):
        loss, aux_loss = self.forward(batch, reduction="none")
        self._inference_outputs += [(loss, batch["task_ids"])]
        return loss, batch["task_ids"]

    def on_test_epoch_end(self, outputs):
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
            os.path.join(self.args.output_dir, "val_loss_by_task.txt"), "a+"
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
        args = self.args
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
