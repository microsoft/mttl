import dataclasses
import json
import os
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Optional, Type

import safetensors
import torch
from pytorch_lightning import seed_everything
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from mttl.arguments import Args, DataArgs, ExpertConfig, ModifierArgs
from mttl.datamodule.base import get_datamodule
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import (
    BaseExpertModel,
    BaseExpertModelConfig,
    ExpertModel,
    ExpertModelConfig,
)
from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler
from mttl.models.hf.callbacks import DownstreamEvalCallback
from mttl.models.hf.trainer import ExpertModelTrainer
from mttl.models.library.expert import Expert, load_expert
from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary
from mttl.models.lightning.callbacks import (
    LiveCheckpointCallback,
    NanoMMLUCallback,
    RougeCallback,
)
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.monitors import get_monitors
from mttl.utils import generate_random_string, rank_zero_only_and_wait, remote_login


def train_experts(
    model_config: BaseExpertModelConfig,
    model_class: Type[BaseExpertModel],
    training_args: ExpertConfig,
):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    logger.info("Args: {}".format(training_args.to_json()))

    remote_login(training_args.remote_token)

    expert_library = None
    if training_args.library_id:

        @rank_zero_only_and_wait(before=False, after=True)
        def create_library(args):
            expert_library = ExpertLibrary.get_expert_library(
                repo_id=args.library_id,
                create=True,
                destination_id=args.destination_library_id,
            )
            return expert_library

        expert_library = create_library(training_args)

    dm = get_datamodule(training_args)

    module = model_class(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    )

    callbacks = []
    if training_args.pipeline_eval_tasks:
        if training_args.pipeline_eval_tasks == "all":
            training_args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        eval = DownstreamEvalCallback(module, training_args)
        callbacks.append(eval)
    else:
        logger.warning(
            "Deactivating downstream eval callback as it is not enabled in the config. Please set `pipeline_eval_tasks`."
        )

    hf_args = training_args.to_hf_training_args()
    trainer: Trainer = ExpertModelTrainer(
        model=module,
        args=hf_args,
        data_collator=dm.collate_fn,
        train_dataset=dm.train_dataset,
        eval_dataset=dm.dev_dataset,
        callbacks=callbacks,
        optimizers=(
            get_optimizer(module, training_args),
            get_scheduler(module, training_args),
        ),
    )

    trainer.train()


if __name__ == "__main__":
    args = ExpertConfig.parse()

    model_config = ExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name,
        modifier_config=args.modifier_config,
    )

    train_experts(model_config, ExpertModel, args)
