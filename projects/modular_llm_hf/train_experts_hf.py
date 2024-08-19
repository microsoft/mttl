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

from mttl.callbacks import LiveCheckpointCallback, NanoMMLUCallback, RougeCallback
from mttl.config import Args, DataArgs, ExpertConfig, ModifierArgs
from mttl.datamodule.base import get_datamodule
from mttl.logging import get_pl_loggers, logger, setup_logging
from mttl.models.expert_configuration import BaseExpertModelConfig
from mttl.models.expert_modeling_base import BaseExpertModel
from mttl.models.expert_modeling_single import (
    SingleExpertModel,
    SingleExpertModelConfig,
)
from mttl.models.library.expert import Expert, load_expert
from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.monitors import get_monitors
from mttl.utils import generate_random_string, rank_zero_only_and_wait, remote_login
from projects.modular_llm_hf.hf_callbacks import DownstreamEvalCallback


class ExpertModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = model(**inputs)
        return (loss, outputs) if return_outputs else loss

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
    )

    trainer.train()


if __name__ == "__main__":
    args = ExpertConfig.parse()

    single_model_config = SingleExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name,
        modifier_config=args.modifier_config,
    )

    train_experts(single_model_config, SingleExpertModel, args)
