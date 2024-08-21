import math
import re
import threading
from collections import defaultdict
from functools import partial
from typing import Dict, List, Union

import torch
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedModel

from mttl.arguments import Args, ExpertConfig, MoEExpertConfig, MultiExpertConfig
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
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelConfig,
    MultiExpertModel,
    MultiExpertModelConfig,
)
from mttl.models.library.expert import Expert, ExpertInfo
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.lightning.base_module import EfficientCheckpointModule
from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.models.modifiers.lora import SkilledLoRAConfig
from mttl.models.modifiers.modify_model import get_modifier_name

torch.set_float32_matmul_precision("high")


class ExpertModule(EfficientCheckpointModule):
    delegate_methods = ["generate", "generation_config", "as_expert"]

    training_config_class = ExpertConfig

    def __getattr__(self, name):
        if name in self.delegate_methods:
            return getattr(self.expert_model, name)
        return super().__getattr__(name)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.tokenizer = kwargs.pop("tokenizer", None)
        self.model_object = kwargs.pop("model_object", None)
        self.save_hyperparameters(kwargs)

        # init the transformer just with the modifier config, this avoids
        # passing the whole training config to the modify_transformer func
        self.training_config = self.training_config_class.fromdict(kwargs)

        # log hyperparameters
        self.accumulate_metrics_batch = defaultdict(list)
        self.best_val_result = None
        self.test_results = []
        self._inference_outputs = []
        self._log_pref = kwargs.get("logging_prefix", "")

        # model setup
        self.setup_expert_model()

    def setup_expert_model(self):
        config = ExpertModelConfig(
            base_model=self.training_config.model,
            task_name=self.training_config.finetune_task_name,
            expert_name=self.training_config.expert_name,
            modifier_config=self.training_config.modifier_config,
        )
        self.expert_model = ExpertModel(
            config,
            model_object=self.model_object,
            device_map=getattr(self.hparams, "device_map", "cpu"),
            attn_implementation=getattr(self.hparams, "attn_implementation", None),
            load_in_4bit=getattr(self.hparams, "load_in_4bit", False),
            load_in_8bit=getattr(self.hparams, "load_in_8bit", False),
        )

    def forward(self, **kwargs):
        outputs = self.expert_model.forward(**kwargs)
        if kwargs.get("return_context", False):
            return outputs[0]
        return outputs[0], outputs[-1]

    def training_step(self, batch, _):
        if "num_options" in batch:
            loss, mc_loss = self.compute_unlikelihood_loss(batch)
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
            loss, _ = self.compute_unlikelihood_loss(batch, reduction="none")
        else:
            loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach(),)]
        return mean_loss

    def get_loss_for_all(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        return loss

    def validation_step(self, batch, batch_idx):
        if "num_options" in batch:
            loss, _ = self.compute_unlikelihood_loss(batch, reduction="none")
        else:
            loss = self.forward(batch, reduction="none")
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

    @property
    def model(self) -> PreTrainedModel:
        """Give access to the underlying HF model object.

        Returns:
            PreTrainedModel: The underlying HF model object.
        """
        return self.expert_model.model

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)

        ckpt["expert_info"] = self.expert_model.as_expert(
            self.training_config
        ).expert_info.asdict()


class MultiExpertModule(ExpertModule):
    """Adds all functions and properties for a multi-expert model."""

    delegate_methods = [
        "add_empty_expert",
        "add_expert_instance",
        "get_expert_instance",
        "load_expert",
        "add_experts_from_dict",
        "add_experts_from_library",
        "set_selector",
        "extract_parameters",
        "get_expert_instance",
        "save_to_library",
    ]
    training_config_class = MultiExpertConfig

    def setup_expert_model(self):
        # config about the routing
        if hasattr(self.hparams, "selector_config"):
            self.selector_config = self.hparams.selector_config
        else:
            self.selector_config = self.training_config.selector_config

        config = MultiExpertModelConfig(
            base_model=self.training_config.model,
            selector_config=self.selector_config,
        )
        self.expert_model = MultiExpertModel(
            config,
            model_object=self.model_object,
            device_map=getattr(self.hparams, "device_map", "cpu"),
            attn_implementation=getattr(self.hparams, "attn_implementation", None),
            load_in_4bit=getattr(self.hparams, "load_in_4bit", False),
            load_in_8bit=getattr(self.hparams, "load_in_8bit", False),
        )

    def __init__(self, selector_config=None, **kwargs):
        super().__init__(**kwargs)


class MoEModule(MultiExpertModule):
    training_config_class = MoEExpertConfig

    def __init__(self, expert_library: ExpertLibrary = None, **kwargs):
        init_from_scratch = kwargs.get("init_from_scratch", False)

        super().__init__(**kwargs)

        if not self.hparams.library_id and expert_library is None or init_from_scratch:
            for i in range(self.hparams.moe_num_experts):
                # Adding a Skilled LoRA with 1 skill.
                exp_config = SkilledLoRAConfig(
                    n_skills=1,
                    modify_layers=self.hparams.modify_layers,
                    modify_modules=self.hparams.modify_modules,
                    lora_alpha=self.hparams.lora_alpha,
                    lora_dropout=self.hparams.lora_dropout,
                    lora_rank=self.hparams.lora_rank,
                    lora_init_b_random=True,
                    n_splits=self.hparams.n_splits,
                    phi_2_align_heads=self.hparams.phi_2_align_heads,
                )
                self.model.add_empty_expert(f"e{i}", exp_config)
            self.moe_num_experts = kwargs["moe_num_experts"]
        else:
            if expert_library is None:
                expert_library = ExpertLibrary.get_expert_library(
                    self.hparams.library_id
                )
            for i, expert in enumerate(sorted(list(expert_library.keys()))):
                self.model.add_expert_instance(
                    expert_library[expert], expert_name=f"e{i}"
                )

            self.moe_num_experts = i + 1

    def training_step(self, batch, _):
        loss, context = self.forward(batch, return_context=True)
        total_loss = loss

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"])

        total_loss = loss.clone()
        routing_gates = context["routing_gates"]

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
