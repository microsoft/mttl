import json
import os
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Type

import torch
from pytorch_lightning import seed_everything
from transformers import Trainer, TrainerCallback, TrainingArguments

from mttl.callbacks import LiveCheckpointCallback, NanoMMLUCallback, RougeCallback
from mttl.config import Args, ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.logging import get_pl_loggers, logger, setup_logging
from mttl.models.expert_model import ExpertModel, MoEModel
from mttl.models.library.expert import Expert, load_expert
from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary
from mttl.models.monitors import get_monitors
from mttl.utils import generate_random_string, rank_zero_only_and_wait, remote_login
from projects.modular_chatbot.hf_callbacks import DownstreamEvalCallback


class ExpertModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = model(**inputs)
        return (loss, outputs) if return_outputs else loss


def train_experts(args: ExpertConfig, model_class: Type[ExpertModel]):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)
    expert_library = None
    if args.library_id:

        @rank_zero_only_and_wait(before=False, after=True)
        def create_library(args):
            expert_library = ExpertLibrary.get_expert_library(
                repo_id=args.library_id,
                create=True,
                destination_id=args.destination_library_id,
            )
            return expert_library

        expert_library = create_library(args)

    loggers = get_pl_loggers(args)

    dm = get_datamodule(args)
    args.n_tasks = len(dm._task_names)
    args.task_names = dm._task_names

    module = model_class(**args.asdict())

    from transformers import TrainerCallback

    callbacks = []

    if args.pipeline_eval_tasks:
        if args.pipeline_eval_tasks == "all":
            args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        eval = DownstreamEvalCallback(module, args)
        callbacks.append(eval)
    else:
        logger.warning(
            "Deactivating downstream eval callback as it is not enabled in the config. Please set `pipeline_eval_tasks`."
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.predict_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=1,
        remove_unused_columns=False,
        eval_strategy="steps" if args.eval_every else "epoch",
        save_strategy="steps" if args.save_every else "epoch",
        load_best_model_at_end=True,
    )

    trainer = ExpertModelTrainer(
        model=module,
        args=training_args,
        data_collator=dm.collate_fn,
        train_dataset=dm.train_dataset,
        eval_dataset=dm.dev_dataset,
        callbacks=callbacks,
    )

    if args.eval_before_training:
        trainer.evaluate()

    trainer.train()


if __name__ == "__main__":
    train_experts(ExpertConfig.parse(), ExpertModel)
