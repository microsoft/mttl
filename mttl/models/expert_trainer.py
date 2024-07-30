import math
import os
from collections import defaultdict
from typing import Any, List, Mapping, Union

import torch
from torch import nn

from mttl import logging
from mttl.logging import logger
from mttl.models.containers.selectors.base import Selector, SelectorConfig
from mttl.models.expert_config import ExpertConfig
from mttl.models.expert_context import InfoContainer
from mttl.models.expert_model import ExpertModel, LoRAMoEModel, MultiExpertModel
from mttl.models.get_scheduler import get_scheduler
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.models.modifiers.lora import SkilledLoRAConfig
from mttl.models.utils import EfficientCheckpointModule, prepare_model_for_kbit_training


class ExpertModelLightningWrapper(EfficientCheckpointModule):
    """Wrapper module for training an expert model with PyTorch Lightning. This encapsulates
    ``training_step``, ``validation_step``, ``test_step`` and all the logs and metrics.

    An ExpertModel instance should be passed to this wrapper with the ``model_object`` argument.
    The rest of the hyperparameters are passed as kwargs and constitutes training arguments
    that will be saved by pytorch lightning as HPs.
    """

    def _update_expert_infos(self, model: ExpertModel):
        self.model.expert_info = ExpertInfo(
            expert_name=self.expert_name,
            expert_task_name=self.finetune_task_name,
            expert_config=self.model.modifier_config,
            training_config=self.training_config,
        )

    def _update_training_config_from_model(
        self, model: ExpertModel, training_config: ExpertConfig
    ):
        """Given that the model has a modifier, update the training config with the modifier config."""
        # Override the model_modifier with the actual modifier used in the model
        modifier_name = model.modifier_config.asdict()["__model_modifier__"]
        training_config.model = model.base_model_name
        training_config.model_modifier = modifier_name
        training_config.update_kwargs(model.modifier_config.asdict(), raise_error=False)

    def __init__(self, model_object: ExpertModel, training_config: ExpertConfig):
        super().__init__()

        # log hyperparameters
        self.accumulate_metrics_batch = defaultdict(list)

        self.model: ExpertModel = model_object
        self._update_training_config_from_model(training_config)

        self.base_model_name = self.model.base_model_name
        self.modifier_config = self.model.modifier_config
        self.training_config = training_config

        self.expert_name = training_config.expert_name
        self.finetune_task_name = training_config.finetune_task_name

        if self.model.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        self.training_config.vocab_size = (
            self.model.model.get_input_embeddings().num_embeddings
        )

        self.best_val_result = None
        self.test_results = []
        self._inference_outputs = []
        self._log_pref = training_config.logging_prefix

        self.save_hyperparameters(self.training_config.asdict())
        self._update_expert_infos(self.model)

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

    def forward(self, batch, **kwargs):
        return self.model.forward(batch, **kwargs)

    def generate(self, batch, **kwargs):
        return self.model.generate(batch, **kwargs)

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

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        **extra_training_kwargs,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt["hyper_parameters"].update(**extra_training_kwargs)

        training_config: ExpertConfig = ExpertConfig.fromdict(ckpt["hyper_parameters"])
        expert_info: ExpertInfo = ExpertInfo.fromdict(ckpt["experts_info"])

        model = ExpertModel(
            training_config.model,
            load_in_4bit=training_config.load_in_4bit,
            load_in_8bit=training_config.load_in_8bit,
            device_map=training_config.device_map,
            expert_info=expert_info,
        )
        wrapper = cls(model, training_config)
        load_result = wrapper.load_state_dict(ckpt["state_dict"], strict=False)
        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"
        return wrapper

    def on_save_checkpoint(self, ckpt):
        """Inject expert info in the checkpoint and removes wrapping from the parameter names."""
        # Delete the non trainable parameters from the model
        super().on_save_checkpoint(ckpt)

        ckpt["experts_info"] = self.model.expert_info.asdict()

    def as_expert(self) -> Expert:
        """Returns the model as an expert instance."""
        expert: Expert = self.model.get_expert_instance()
        # inject trainer information into the expert
        expert.expert_info = self.expert_info
        return expert


class MultiExpertModelLightningWrapper(ExpertModelLightningWrapper):
    def _update_training_config_from_model(
        self, model: ExpertModel, training_config: ExpertConfig
    ):
        # update the training config with the modifier config
        training_config.model_modifier = None
        training_config.model = model.base_model_name

        if self.model.selector_config:
            training_config.router_selector = Selector.get_name_by_config_class(
                model.selector_config
            )
            training_config.update_kwargs(
                model.selector_config.asdict(), raise_error=False
            )

    def _update_expert_infos(self, model: MultiExpertModel):
        # update the expert info for each expert with the training config
        for expert_info in self.expert_info:
            expert_info.training_config = self.training_config

    @property
    def expert_info(self) -> List[ExpertInfo]:
        self.model: MultiExpertModel
        return [
            self.model.get_expert_instance(n).expert_info
            for n in self.model.experts_names
        ]

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        **extra_training_kwargs,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt["hyper_parameters"].update(**extra_training_kwargs)

        training_config: ExpertConfig = ExpertConfig.fromdict(ckpt["hyper_parameters"])
        selector_config: SelectorConfig = SelectorConfig.fromdict(
            ckpt["selector_config"]
        )
        experts_info: List[ExpertInfo] = [
            ExpertInfo.fromdict(e) for e in ckpt["experts_info"]
        ]

        model = MultiExpertModel(
            training_config.model,
            load_in_4bit=training_config.load_in_4bit,
            load_in_8bit=training_config.load_in_8bit,
            device_map=training_config.device_map,
            experts_info=experts_info,
            selector_config=selector_config,
        )
        wrapper = cls(model, training_config)
        load_result = wrapper.load_state_dict(ckpt["state_dict"], strict=False)
        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"
        return wrapper

    def on_save_checkpoint(self, ckpt):
        # Delete the non trainable parameters from the model
        EfficientCheckpointModule.on_save_checkpoint(self, ckpt)

        ckpt["experts_info"] = [e.asdict() for e in self.expert_info]
        ckpt["selector_config"] = self.model.selector_config.asdict()


class LoRAMoELightningWrapper(MultiExpertModelLightningWrapper):
    def _update_training_config_from_model(self, model: LoRAMoEModel):
        # update the training config with the modifier config
        self.training_config.model_modifier = None
        self.training_config.router_selector = SelectorConfig.get_name_by_config_class(
            model.selector_config
        )
        self.training_config.update_kwargs(
            model.selector_config.asdict(), raise_error=False
        )
        # inject library id if it exists in the model
        self.training_config.library_id = model.expert_library_id

    def on_save_checkpoint(self, ckpt):
        # Delete the non trainable parameters from the model
        model: LoRAMoEModel = self.model

        EfficientCheckpointModule.on_save_checkpoint(self, ckpt)

        ckpt["experts_info"] = [e.asdict() for e in self.expert_info]
        ckpt["experts_library_id"] = model.expert_library_id
        ckpt["selector_config"] = self.model.selector_config.asdict()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        **extra_training_kwargs,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        ckpt["hyper_parameters"].update(**extra_training_kwargs)

        training_config: ExpertConfig = ExpertConfig.fromdict(ckpt["hyper_parameters"])
        selector_config: SelectorConfig = SelectorConfig.fromdict(
            ckpt["selector_config"]
        )
        experts_library_id = ckpt.get("experts_library_id", None)
        experts_info: List[ExpertInfo] = [
            ExpertInfo.fromdict(e) for e in ckpt["experts_info"]
        ]

        # reload the model from checkpoint using experts info
        model = LoRAMoEModel(
            training_config.model,
            selector_config=selector_config,
            experts_library_id=experts_library_id,
            experts_info=experts_info if experts_library_id is None else None,
            load_in_4bit=training_config.load_in_4bit,
            load_in_8bit=training_config.load_in_8bit,
            device_map=training_config.device_map,
        )
        wrapper = cls(model, training_config)
        load_result = wrapper.load_state_dict(ckpt["state_dict"], strict=False)
        assert (
            len(load_result.unexpected_keys) == 0
        ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"
        return wrapper

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
                normalized_entropy = entropy_of_route / math.log(
                    self.selector_config.num_experts
                )
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
