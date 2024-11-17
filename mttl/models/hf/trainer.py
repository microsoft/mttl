import os
from typing import Dict, Optional

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME, TrainingArguments

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.logging import logger
from mttl.models.get_optimizer import get_optimizer_and_scheduler

MTTL_ARGS_NAME = "mttl_args.bin"


class ExpertModelTrainer(Trainer):
    """Generic HF trainer for expert models.

    - Adds support for custom datamodule, optimizer, and scheduler.
    - Adds support for logging other metrics than loss, can use `log_metrics` method to do so, for example during `compute_loss`.
    """

    def __init__(self, model: torch.nn.Module, args: ExpertConfig, **kwargs):
        self.mttl_args: ExpertConfig = args

        args: TrainingArguments = args.to_hf_training_args()
        logger.info(args)

        if kwargs.get("train_dataset") is None:
            logger.info("Initializing datamodule and storing it into Trainer.")
            self.dm = get_datamodule(self.mttl_args)

            kwargs["train_dataset"] = self.dm.train_dataset
            kwargs["eval_dataset"] = self.dm.dev_dataset
            kwargs["data_collator"] = self.dm.collate_fn
            kwargs["tokenizer"] = self.dm.tokenizer
        else:
            self.dm = None

        if kwargs.get("optimizers") is None:
            logger.info("Initializing custom non-HF optimizer and scheduler.")

            (optimizer, scheduler), _ = get_optimizer_and_scheduler(
                model,
                self.mttl_args,
                num_train_examples=len(kwargs["train_dataset"]),
                no_decay=["bias", "LayerNorm.weight"],
            )
            kwargs["optimizers"] = (optimizer, scheduler)

        self._extra_logs = {}
        super().__init__(model=model, args=args, **kwargs)

    @property
    def test_dataset(self):
        if self.dm is not None:
            return self.dm.test_dataset

    def log_metrics(self, metrics: Dict[str, float]):
        self._extra_logs.update(metrics)

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
        ):
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.detach().item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
            logs["learning_rate"] = self._get_learning_rate()

            for key, value in self._extra_logs.items():
                loss_scalar = self._nested_gather(value).mean().item()
                logs[key] = round(
                    loss_scalar
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )

            self._extra_logs.clear()
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def compute_loss(self, model, batch, return_outputs=False):
        outputs = model(**batch)
        return (outputs.loss, outputs.logits) if return_outputs else outputs.loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=self.args.save_safetensors,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.mttl_args.asdict(), os.path.join(output_dir, MTTL_ARGS_NAME))


class LMTrainer(ExpertModelTrainer):
    """Standard next-token prediction objective"""

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return (outputs.loss, outputs.logits) if return_outputs else outputs.loss
