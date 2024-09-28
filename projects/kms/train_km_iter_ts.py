import copy
import gc
import math
import os
import shutil
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import vllm
from lightning_fabric import seed_everything
from transformers import AutoModelForCausalLM
from transformers.trainer import (
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    ExportableState,
    OptimizerNames,
    ParallelMode,
    TrainerState,
    TrainingArguments,
    TrainOutput,
    get_model_param_count,
    is_accelerate_available,
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    speed_metrics,
)

from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.models.get_scheduler import get_scheduler

if is_accelerate_available():
    from accelerate import Accelerator
    from accelerate import __version__ as accelerate_version
    from accelerate import skip_first_batches
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )


if is_apex_available():
    from apex import amp


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

from datasets import DatasetDict
from km_datamodule import KMDatasetConfig, KMDatasetModule
from torch import distributed as dist
from torch import functional as F
from torch import nn
from torch.utils.data import DataLoader
from train_km import DCDTrainer, LMTrainer
from train_km_iter import (
    IterativeDCDTrainer,
    IterKMArguments,
    TextDatamodule,
    TextDatasetConfig,
)

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import DataModule, DatasetConfig, get_datamodule
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelConfig,
    MultiExpertModel,
    MultiExpertModelConfig,
)
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.utils import remote_login


class TeacherStudentDCDTrainer(IterativeDCDTrainer):
    """There are two models:

    - The teacher model has access to the document information and needs to come up with summaries
      that the student cannot really model well. To do so, the teacher model is trained with RL
      to generate summaries and is rewarded by the student model's perplexity on the generated summaries.

      R(summary) = log P_student(summary)

      It'd be good to use PPO for this, such that the teacher is close to the initial reference model.

    - The student on the other hand is trained to predict the next token in the summary with next-word prediction loss.

    Both student and teacher model are two adapters on top of the same base model.
    """

    def __init__(self, model, args, **kwargs):
        super().__init__(model, args, **kwargs)

        self.reference_model = "reference_model"
        self.teacher_model = "teacher_model"
        self.student_model = "student_model"

    def compute_loss(self, model, inputs, return_outputs=False):
        # document + small task prompt + task output (e.g. summary, or question and answer)
        batch_size = inputs["input_ids"].size(0)

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]

        # small task prompt + task output (e.g. summary, or question and answer)
        nc_input_ids = inputs["nc_input_ids"]
        nc_labels = inputs["nc_labels"]
        nc_attention_mask = inputs["nc_attention_mask"]

        with torch.no_grad():
            reference_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_names=[self.reference_model] * batch_size,
                output_hidden_states=True,
                return_dict=True,
            )

        teacher_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_names=[self.teacher_model] * batch_size,
            output_hidden_states=True,
            return_dict=True,
        )

        student_outputs = model(
            input_ids=nc_input_ids,
            attention_mask=nc_attention_mask,
            task_names=[self.student_model] * batch_size,
            output_hidden_states=True,
            return_dict=True,
        )

        labels_mask = labels != -100
        nc_labels_mask = nc_labels != -100
        reference_logits = reference_outputs.logits[labels_mask, ...]
        teacher_logits = teacher_outputs.logits[labels_mask, ...]
        student_logits = student_outputs.logits[nc_labels_mask, ...]

        #### Student Loss ####
        reference_probs = torch.softmax(reference_logits, dim=-1)
        student_logprobs = torch.log_softmax(student_logits, dim=-1)
        student_kl = self.kl_loss(student_logprobs, reference_probs)

        losses = []
        for student_states, reference_states in zip(
            student_outputs.hidden_states, reference_outputs.hidden_states
        ):
            # actual states are the hidden states of the student model
            student_states = student_states[nc_labels_mask, ...]
            reference_states = reference_states[labels_mask, ...]

            # Calculate mean magnitude of target states
            mean = torch.sum(torch.abs(reference_states)) / student_states.size(-1)

            losses.append(
                # Loss is the mean abs difference between target and predicted states,
                # normalised by mean magnitude of target states
                torch.sum(torch.abs(student_states - reference_states))
                / (mean * np.prod(reference_states.shape))
            )

        student_hidden_loss = torch.mean(torch.stack(losses))
        student_kl_loss = student_kl.sum(dim=-1).mean()
        student_loss = student_kl_loss + student_hidden_loss

        #### Teacher Loss ####
        reference_logprobs = torch.log_softmax(reference_logits, dim=-1)
        teacher_logprobs = torch.log_softmax(teacher_logits, dim=-1)

        reference_nll = -torch.gather(
            reference_logprobs, 1, labels[labels_mask].unsqueeze(1)
        ).view(-1)
        teacher_nll = -torch.gather(
            teacher_logprobs, 1, labels[labels_mask].unsqueeze(1)
        ).view(-1)
        student_nll = -torch.gather(
            student_logprobs, 1, nc_labels[nc_labels_mask].unsqueeze(1)
        ).view(-1)

        rewards = student_nll - reference_nll
        rewards = torch.clamp(rewards - rewards.mean(), -1.0, 1.0)

        # compute teacher loss only on top_k tokens
        teacher_loss = (rewards.detach() * teacher_nll).mean()
        teacher_kl = torch.kl_div(reference_logprobs, teacher_logprobs, log_target=True)

        if teacher_kl.ndim == 2:
            teacher_kl = teacher_kl.sum(dim=-1)

        teacher_kl = teacher_kl.mean()
        teacher_loss = teacher_loss + teacher_kl

        return (
            (teacher_loss + student_loss, student_logits)
            if return_outputs
            else teacher_loss + student_loss
        )

    def generate_epoch_data(self, train_dataset):
        # Clean up temporary directory
        import shutil

        from dataset_augmenter import DatasetAugmenter

        from mttl.models.utils import model_loader_helper

        shutil.rmtree(self.temp_directory, ignore_errors=True)
        os.makedirs(self.temp_directory, exist_ok=True)

        device = self.model.device

        # we need to save the model in the temp directory, this moves it to CPU so that VLLM
        # doesn't complain, one alternative is to use 2 gpus, one for generation, one for training
        self.model.merge_and_save_base_model(
            self.temp_directory, expert_name="teacher_model", device="cpu"
        )

        # Generate a set of prompts
        augmenter = DatasetAugmenter(
            self.temp_directory,
            block_size=2048,
            max_continuation_length=768,
            num_generations=16,
            generation_top_p=0.95,
        )
        augmenter.add_task("summary")
        augmented_dataset = augmenter.augment(dataset=train_dataset)

        # Force garbage collection
        del augmenter
        gc.collect()
        torch.cuda.empty_cache()
        self.model.to(device)

        return DatasetDict({"train": augmented_dataset})


def train_km(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    model_config = MultiExpertModelConfig(
        base_model=args.model,
    )

    model = MultiExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    )
    for expert_name in ["teacher_model", "student_model", "reference_model"]:
        model.add_empty_expert(expert_name, expert_config=args.modifier_config)

    # set the default expert to the student model
    model.set_default_expert("student_model")

    callbacks = []
    if training_args.nqa_dataset is not None:
        # load the NQA callback to monitor zero-shot performance
        from nqa_callback import NQAZeroShotCallback

        data_args = copy.deepcopy(training_args)
        data_args.dataset = training_args.nqa_dataset
        callback = NQAZeroShotCallback(model, data_args)
        callbacks.append(callback)

    trainer = TeacherStudentDCDTrainer(
        model=model,
        args=training_args,
        callbacks=callbacks,
    )

    trainer.train()

    # Get the best checkpoint
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        logger.info("Best model checkpoint: %s", best_model_path)


if __name__ == "__main__":
    args = IterKMArguments.parse()

    train_km(args)
