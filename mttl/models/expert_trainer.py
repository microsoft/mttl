import math
import os
from collections import defaultdict
from typing import Any, Mapping

import torch
from torch import nn

from mttl.logging import logger
from mttl.models.containers.selectors.base import SelectorConfig
from mttl.models.expert_config import ExpertConfig
from mttl.models.expert_context import InfoContainer
from mttl.models.expert_model import MoEModel, SingleExpertModel
from mttl.models.get_scheduler import get_scheduler_with_args
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.lora import SkilledLoRAConfig
from mttl.models.utils import EfficientCheckpointModule, prepare_model_for_kbit_training


class SingleExpertModelLightningWrapper(EfficientCheckpointModule):
    """Module for training an expert model with PyTorch Lightning."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = kwargs.pop("tokenizer", None)
        self.model = kwargs.pop("model_object", None)

        # log hyperparameters
        self.save_hyperparameters(kwargs)
        self.accumulate_metrics_batch = defaultdict(list)

        # rebuild the training config, a bit cumbersome, but that's life
        self.training_config: ExpertConfig = ExpertConfig.fromdict(kwargs)

        if self.model is None:
            modifier_config = ModifierConfig.from_training_config(self.training_config)
            self.base_model_name = kwargs.get("model", None)
            self.model = SingleExpertModel(
                self.base_model_name,
                modifier_config=modifier_config,
                **kwargs,
            )
        else:
            self.base_model_name = self.model.base_model_name

        self.modifier_config = self.model.modifier_config

        if self.model.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        self.training_config.vocab_size = (
            self.model.get_input_embeddings().num_embeddings
        )

        self.test_results = []
        self.best_val_result = None
        self._inference_outputs = []
        self._log_pref = kwargs.get("logging_prefix", "")

    def training_step(self, batch, _):
        loss = self.model.forward(batch)
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
        loss = self.model.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach(),)]
        return mean_loss

    def get_loss_for_all(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach(),)]
        return mean_loss

    def log_loss(self, split="val"):
        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)
        self._inference_outputs.clear()
        self.log(
            f"{self._log_pref}{split}/loss",
            losses.mean(),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
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

    def on_save_checkpoint(self, ckpt):
        """Inject expert info in the checkpoint."""
        # inject expert info in the expert checkpoint
        ckpt["expert_info"] = self.expert_info.asdict()
        ckpt["modifier_config"] = self.modifier_config.asdict()

    @property
    def expert_info(self) -> ExpertInfo:
        # inject expert info in the expert checkpoint
        expert_info = ExpertInfo(
            expert_name=self.hparams.expert_name,
            expert_task_name=self.hparams.finetune_task_name,
            expert_config=self.modifier_config,
            training_config=self.training_config,
        )
        return expert_info

    def as_expert(self) -> Expert:
        """Returns the model as an expert instance."""
        if not isinstance(self.model, SingleExpertModel):
            raise ValueError(
                "Only SingleExpertModel can be converted to a single expert."
            )

        state_dict = self.model.state_dict()
        self._delete_non_trainable_params(state_dict)

        # to use as an expert, we need to remove a `model.` prefix
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

        # inject expert info in the expert checkpoint
        return Expert(
            expert_info=self.expert_info,
            expert_weights=state_dict,
        )


class LoRAMoELightningWrapper(SingleExpertModelLightningWrapper):
    def __init__(self, **kwargs):
        self.tokenizer = kwargs.pop("tokenizer", None)
        self.model = kwargs.pop("model_object", None)

        # log hyperparameters
        self.save_hyperparameters(kwargs)
        self.accumulate_metrics_batch = defaultdict(list)

        # rebuild the training config, a bit cumbersome, but that's life
        self.training_config: ExpertConfig = ExpertConfig.fromdict(kwargs)

        if self.model is None:
            modifier_config = SkilledLoRAConfig.from_training_config(
                self.training_config
            )
            selector_config = SelectorConfig.from_training_config(self.training_config)
            self.base_model_name = kwargs.get("model", None)
            self.model = MoEModel(
                self.base_model_name,
                modifier_config=modifier_config,
                selector_config=selector_config,
                **kwargs,
            )
        else:
            self.base_model_name = self.model.base_model_name

        self.selector_config = self.model.selector_config
        self.modifier_config = self.model.modifier_config

        if self.model.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        self.training_config.vocab_size = (
            self.model.get_input_embeddings().num_embeddings
        )

        self.test_results = []
        self.best_val_result = None
        self._inference_outputs = []
        self._log_pref = kwargs.get("logging_prefix", "")

    def training_step(self, batch, _):
        loss = super().training_step(batch, _)
        total_loss = loss.clone()

        routing_gates = InfoContainer.get().routing_gates
        if routing_gates:
            num = 0.0
            entropy_of_avg = 0.0
            entropy_of_route = 0.0

            for values in routing_gates:
                # compute MI loss
                values = values.to(torch.float32)
                values = values.view(-1, values.shape[-1])
                probs = torch.softmax(values, -1)
                entropy_of_avg += -(
                    probs.mean(0) * torch.log(probs.mean(0) + 1e-6)
                ).sum(-1)
                entropy_of_route += -(probs * torch.log(probs + 1e-6)).sum(-1).mean(0)
                num += 1.0

            entropy_of_avg = entropy_of_avg / num
            entropy_of_route = entropy_of_route / num
            mi_loss = -entropy_of_avg + entropy_of_route

            self.log(
                f"{self._log_pref}train/entropy_of_route",
                entropy_of_route,
                on_step=True,
                prog_bar=True,
            )
            self.log(
                f"{self._log_pref}train/entropy_of_avg",
                entropy_of_avg,
                on_step=True,
                prog_bar=True,
            )
            self.log(
                f"{self._log_pref}train/mi_loss",
                mi_loss,
                on_step=True,
                prog_bar=True,
            )

            if self.hparams.moe_ent_reg > 0.0:
                total_loss += self.hparams.moe_ent_reg * mi_loss

            elif self.hparams.moe_ent_free_bits > 0.0:
                normalized_entropy = entropy_of_route / math.log(self.moe_num_experts)
                total_loss += (
                    (1.0 - normalized_entropy) >= self.hparams.moe_ent_free_bits
                ) * -entropy_of_route

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])
        return total_loss

    def on_save_checkpoint(self, ckpt):
        """Inject expert info in the checkpoint."""
        ckpt["expert_info"] = self.expert_info.asdict()
        ckpt["modifier_config"] = self.modifier_config.asdict()
        ckpt["selector_config"] = self.selector_config.asdict()
