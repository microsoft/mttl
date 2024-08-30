import math
from collections import defaultdict
from typing import List

import torch
from transformers import PreTrainedModel

from mttl.arguments import ExpertConfig, MoEExpertConfig, MultiExpertConfig
from mttl.models.containers.selectors.base import AutoSelectorConfig
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelConfig,
    MoEModel,
    MoEModelConfig,
    MultiExpertMixin,
    MultiExpertModel,
    MultiExpertModelConfig,
)
from mttl.models.library.expert import ExpertInfo
from mttl.models.lightning.base_module import LightningEfficientCheckpoint
from mttl.models.utils import compute_loglike_loss

torch.set_float32_matmul_precision("high")


class LightningTrainingMixin:
    @property
    def _log_pref(self):
        return getattr(self.hparams, "logging_prefix", "")

    @property
    def inference_outputs(self):
        if not hasattr(self, "_inference_outputs"):
            self._inference_outputs = []
        return self._inference_outputs

    @property
    def best_val_result(self):
        if not hasattr(self, "_best_val_result"):
            self._best_val_result = None
        return self._best_val_result

    @best_val_result.setter
    def best_val_result(self, value):
        self._best_val_result = value

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def training_step(self, batch, _):
        outputs = self.forward(**batch)
        loss = outputs.loss
        total_loss = loss

        self.log(f"{self._log_pref}train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            f"{self._log_pref}train/total_loss", total_loss, on_step=True, prog_bar=True
        )

        for i, pg in enumerate(self.optimizers().optimizer.param_groups):
            self.log(f"train/lr_{i}", pg["lr"], prog_bar=True)
        return total_loss

    def on_validation_epoch_end(self) -> None:
        self.log_loss(split="val")
        self.inference_outputs.clear()

    def on_test_epoch_end(self) -> None:
        self.log_loss(split="test")
        self.inference_outputs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = compute_loglike_loss(outputs.logits, batch["labels"], reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        return mean_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        loss = compute_loglike_loss(outputs.logits, batch["labels"], reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        return mean_loss

    def log_loss(self, split="val"):
        outputs = self.inference_outputs
        losses = torch.cat([out[0] for out in outputs], 0)
        self.inference_outputs.clear()

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


class ExpertModule(LightningTrainingMixin, LightningEfficientCheckpoint):
    def __init__(self, model_object=None, **kwargs):
        LightningEfficientCheckpoint.__init__(self, model_object=model_object, **kwargs)

        self.training_config = ExpertConfig.fromdict(kwargs)

        config = ExpertModelConfig(
            base_model=self.training_config.model,
            task_name=self.training_config.finetune_task_name,
            expert_name=self.training_config.expert_name,
            modifier_config=self.training_config.modifier_config,
        )
        self.model = ExpertModel(
            config,
            model_object=self.model_object,
            device_map=getattr(self.hparams, "device_map", "cpu"),
            precision=getattr(self.hparams, "precision", "bf16"),
            attn_implementation=getattr(self.hparams, "attn_implementation", None),
            load_in_4bit=getattr(self.hparams, "load_in_4bit", False),
            load_in_8bit=getattr(self.hparams, "load_in_8bit", False),
        )

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)

        ckpt["expert_info"] = self.as_expert(self.training_config).expert_info.asdict()


class MultiExpertModule(LightningTrainingMixin, LightningEfficientCheckpoint):
    def __init__(
        self,
        model_object: PreTrainedModel = None,
        expert_infos: List[ExpertInfo] = None,
        selector_config: AutoSelectorConfig = None,
        **kwargs,
    ):
        LightningEfficientCheckpoint.__init__(self, model_object=model_object, **kwargs)

        self.training_config = MultiExpertConfig.fromdict(kwargs)

        config = MultiExpertModelConfig(
            base_model=self.training_config.model,
            selector_config=selector_config or self.training_config.selector_config,
            expert_infos=expert_infos,
        )
        self.model = MultiExpertModel(
            config,
            model_object=self.model_object,
            device_map=getattr(self.hparams, "device_map", "cpu"),
            precision=getattr(self.hparams, "precision", "bf16"),
            attn_implementation=getattr(self.hparams, "attn_implementation", None),
            load_in_4bit=getattr(self.hparams, "load_in_4bit", False),
            load_in_8bit=getattr(self.hparams, "load_in_8bit", False),
        )

    def add_expert_instance(self, *args, **kwargs):
        return self.model.add_expert_instance(*args, **kwargs)

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)

        ckpt["expert_infos"] = [
            self.model.get_expert_instance(n).expert_info.asdict()
            for n in self.model.experts_names
        ]
        ckpt["selector_config"] = self.model.selector_config.asdict()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        **model_kwargs,
    ):
        from mttl.datamodule.utils import get_tokenizer_with_args

        tokenizer = model_kwargs.get("tokenizer", None)
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        ckpt["hyper_parameters"].update(**model_kwargs)

        expert_info = None
        selector_config = None

        if "expert_info" in ckpt:
            expert_info = [ExpertInfo.fromdict(k) for k in ckpt["expert_info"]]
        if "selector_config" in ckpt:
            selector_config = AutoSelectorConfig.fromdict(ckpt["selector_config"])

        model = cls(
            expert_info=expert_info,
            selector_config=selector_config,
            **ckpt["hyper_parameters"],
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model


class MoEModule(LightningEfficientCheckpoint, LightningTrainingMixin):
    def __init__(
        self,
        model_object: PreTrainedModel = None,
        selector_config: AutoSelectorConfig = None,
        **kwargs,
    ):
        LightningEfficientCheckpoint.__init__(self, model_object=model_object, **kwargs)

        self.training_config = MoEExpertConfig.fromdict(kwargs)

        config = MoEModelConfig(
            base_model=self.training_config.model,
            moe_num_experts=self.training_config.moe_num_experts,
            modifier_config=self.training_config.modifier_config,
            selector_config=selector_config or self.training_config.selector_config,
        )
        self.model = MoEModel(
            config,
            model_object=self.model_object,
            device_map=getattr(self.hparams, "device_map", "cpu"),
            precision=getattr(self.hparams, "precision", "bf16"),
            attn_implementation=getattr(self.hparams, "attn_implementation", None),
            load_in_4bit=getattr(self.hparams, "load_in_4bit", False),
            load_in_8bit=getattr(self.hparams, "load_in_8bit", False),
        )

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)
        ckpt["selector_config"] = self.model.selector_config.asdict()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        **model_kwargs,
    ):
        from mttl.datamodule.utils import get_tokenizer_with_args

        tokenizer = model_kwargs.get("tokenizer", None)
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        ckpt["hyper_parameters"].update(**model_kwargs)

        selector_config = None

        if "selector_config" in ckpt:
            selector_config = AutoSelectorConfig.fromdict(ckpt["selector_config"])

        model = cls(
            selector_config=selector_config,
            **ckpt["hyper_parameters"],
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model

    def training_step(self, batch, _):
        output, context = self.forward(**batch, return_context=True)
        loss = output.loss
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
