import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from pyparsing import Any
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME, TrainingArguments

from mttl.arguments import ExpertConfig
from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler


class ExpertModelTrainer(Trainer):
    """Generic HF trainer for expert models."""

    def __init__(self, model, args, **kwargs):
        args: ExpertConfig = args

        if "optimizers" in kwargs:
            raise ValueError("`optimizers` not supported!")

        optimizer = get_optimizer(model, args)[0]
        scheduler = get_scheduler(optimizer, args)

        hf_args: TrainingArguments = args.to_hf_training_args()

        super().__init__(
            model=model, args=hf_args, optimizers=(optimizer, scheduler), **kwargs
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
