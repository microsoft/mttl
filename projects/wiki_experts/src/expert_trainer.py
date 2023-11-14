import torch
from collections import defaultdict
from torch import nn
from mttl.models.llama_patch import replace_attn_with_flash_attn
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.routing import RoutingInfo
from transformers import AutoModelForCausalLM, LlamaForCausalLM

from mttl.models.utils import (
    EfficientCheckpointModule,
    prepare_model_for_kbit_training,
)
from mttl.utils import logger


class ExpertTrainer(EfficientCheckpointModule):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(**kwargs)

        # log hyperparameters
        self.save_hyperparameters(ignore=["tokenizer", "model_object"])

        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.model: AutoModelForCausalLM = None
        self.accumulate_metrics_batch = defaultdict(list)

        if "llama" in self.hparams.model:
            model_object = LlamaForCausalLM.from_pretrained(
                self.hparams.model,
                load_in_8bit=self.hparams.load_in_8bit,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            model_object = AutoModelForCausalLM.from_pretrained(self.hparams.model)

        if self.hparams.load_in_8bit:
            model_object = prepare_model_for_kbit_training(model_object)

        self.model = modify_transformer(model_object, self.hparams)

        # replace w flash attn!
        replace_attn_with_flash_attn(self.model)

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

    def training_step(self, batch, _):
        loss = self.forward(batch)
        total_loss = loss

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True)

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
        loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach().cpu(),)]
        return mean_loss

    def get_loss_for_all(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, reduction="none")
        mean_loss = loss.sum() / loss.shape[0]
        self._inference_outputs += [(loss.detach().cpu(),)]
        return mean_loss

    def log_loss(self, split="val"):
        outputs = self._inference_outputs
        losses = torch.cat([out[0] for out in outputs])
        self._inference_outputs.clear()
        self.log(f"{split}/loss", losses.mean(), on_epoch=True, prog_bar=True)

        # log also the best val/loss sofar
        if split == "val":
            if self.best_val_result is None:
                self.best_val_result = losses.mean()
            else:
                if losses.mean() < self.best_val_result:
                    self.best_val_result = losses.mean()
                    self.log(
                        f"{split}/best_loss",
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
        if hasattr(self.model, "task_id_container"):
            self.model.task_id_container["routing_infos"] = RoutingInfo.from_batch(
                batch
            )

        generations = self.model.generate(
            inputs=batch["input_ids"], attention_mask=batch["attention_mask"], **kwargs
        )
        return generations
