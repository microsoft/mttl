import torch
import numpy as np
from collections import defaultdict
from torch import nn
from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.routing import RoutingInfo
from transformers import AutoModelForCausalLM

from mttl.models.modifiers.expert_containers.module_graph import ExpertInfo
from mttl.models.utils import (
    EfficientCheckpointModule,
    prepare_model_for_kbit_training,
)
from mttl.models.modifiers.expert_containers.module_graph import Expert
from projects.wiki_experts.src.config import ExpertConfig
from mttl.models.modifiers.expert_containers.selectors import SelectorConfig


torch.set_float32_matmul_precision("high")


class ExpertTrainer(EfficientCheckpointModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = kwargs.pop("tokenizer", None)
        model_object = kwargs.pop("model_object", None)

        # log hyperparameters
        self.save_hyperparameters(kwargs)

        self.model: AutoModelForCausalLM = None
        self.accumulate_metrics_batch = defaultdict(list)

        if model_object is None:
            from mttl.models.utils import model_loader_helper

            model_object = model_loader_helper(
                self.hparams.model,
                load_in_8bit=self.hparams.load_in_8bit,
                device_map=getattr(self.hparams, "device_map", "cpu"),
            )

        if self.hparams.load_in_8bit:
            model_object = prepare_model_for_kbit_training(model_object)

        # rebuild the training config, a bit cumbersome, but that's life
        self.training_config = ExpertConfig.fromdict(kwargs)
        self.training_config.vocab_size = (
            model_object.get_input_embeddings().num_embeddings
        )

        # init the transformer just with the modifier config, this avoids
        # passing the whole training config to the modify_transformer func
        self.modifier_config = ModifierConfig.from_training_config(self.training_config)
        # config about the routing
        self.routing_config = SelectorConfig.from_training_config(self.training_config)

        self.model = modify_transformer(model_object, self.modifier_config)

        # replace w flash attn!
        replace_attn_with_flash_attn(self.model)

        self.test_results = []
        self.best_val_result = None
        self._inference_outputs = []
        self._log_pref = kwargs.get("logging_prefix", "")

    def set_routing_infos(self, batch, generate=False):
        self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(batch)

    def forward_unlikelihood(self, batch, reduction="mean"):
        # compute lm losses for all the options
        loss = self.forward(batch, reduction="none")

        num = 0
        lm_loss, mc_loss = [], []
        for i, num_options in enumerate(batch["num_options"]):
            loss_slice = loss[num : num + num_options]
            lm_loss.append(loss_slice[batch["labels_index"][i]])
            mc_loss.append(
                -torch.nn.functional.log_softmax(-loss_slice, dim=0)[
                    batch["labels_index"][i]
                ]
            )
            num += num_options

        lm_loss = torch.stack(lm_loss)
        mc_loss = torch.stack(mc_loss)
        if reduction == "mean":
            lm_loss = lm_loss.mean()
            mc_loss = mc_loss.mean()
        return lm_loss, mc_loss

    def forward(self, batch, reduction="mean"):
        input_ids, labels = batch["input_ids"], batch["labels"]

        self.set_routing_infos(batch)

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

    def training_step(self, batch, _):
        if "num_options" in batch:
            loss, mc_loss = self.forward_unlikelihood(batch)
            self.log(
                f"{self._log_pref}train/mc_loss", loss, on_step=True, prog_bar=True
            )
            total_loss = loss + mc_loss
        else:
            loss = self.forward(batch)
            total_loss = loss

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss

    def on_validation_epoch_start(self) -> None:
        self._inference_outputs.clear()

    def on_test_epoch_start(self) -> None:
        self._inference_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.log_loss(split="val")

    def on_test_epoch_end(self) -> None:
        self.log_loss(split="test")

    def test_step(self, batch, batch_idx):
        if "num_options" in batch:
            loss, _ = self.forward_unlikelihood(batch, reduction="none")
        else:
            loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach().cpu(),)]
        return mean_loss

    def get_loss_for_all(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        return loss

    def validation_step(self, batch, batch_idx):
        if "num_options" in batch:
            loss, _ = self.forward_unlikelihood(batch, reduction="none")
        else:
            loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach().cpu(),)]
        return mean_loss

    def log_loss(self, split="val"):
        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)
        self._inference_outputs.clear()
        self.log(
            f"{self._log_pref}{split}/loss", losses.mean(), on_epoch=True, prog_bar=True
        )

        # log also the best val/loss sofar
        if split == "val":
            if self.best_val_result is None:
                self.best_val_result = losses.mean()
            else:
                if losses.mean() < self.best_val_result:
                    self.best_val_result = losses.mean()
                    self.log(
                        f"{self._log_pref}{split}/best_loss",
                        losses.mean(),
                        on_epoch=True,
                        prog_bar=True,
                    )

    @property
    def generation_config(self):
        return self.model.generation_config

    def generate(
        self,
        batch,
        **kwargs,
    ):
        self.set_routing_infos(batch, generate=True)

        generations = self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return generations

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)

        # inject expert info in the expert checkpoint
        expert_info = ExpertInfo(
            expert_name=self.hparams.expert_name,
            expert_task_name=self.hparams.finetune_task_name,
            expert_config=self.modifier_config,
            training_config=self.training_config,
        )
        ckpt["expert_info"] = expert_info.asdict()
