import copy
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from nqa_datamodule import NQADatamodule  # noqa: F401
from nqa_evaluator import NQAZeroShotEvaluator
from simple_utils import SimpleLogger, dcd_loss, do_evaluation
from tqdm import tqdm

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import DataModule, DatasetConfig, get_datamodule
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig, disable_modifiers
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device
from mttl.utils import create_library, remote_login, upload_library


@dataclass
class TextDatasetConfig(DatasetConfig):
    text_column: str = "text"
    task_name_field: str = "document_id"


@DataModule.register("text_dataset", config_cls=TextDatasetConfig)
class TextDatamodule(DataModule):
    """Just a datamodule that loads a dataset with a 'text' column."""

    def collate_fn(self, examples):
        return None

    def setup_dataset(self):
        from mttl.models.library.dataset_library import DatasetLibrary

        dataset = DatasetLibrary.pull_dataset(self.config.dataset)

        if self.config.text_column not in dataset["train"].column_names:
            raise ValueError(f"The dataset must contain a `{self.config.text}` column")

        (
            self._task_names,
            self._task_to_id,
            self.train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset,
            self.config.task_name_field,
            self.config.finetune_task_name,
        )

        self.dev_dataset = None
        self.test_dataset = None


def create_datamodule(training_args, dataset_path):
    args_copy = copy.deepcopy(training_args)
    args_copy.dataset = "local://" + dataset_path
    args_copy.dataset_type = "dcd_km"
    return get_datamodule(args_copy)


def create_synthetic_data_for_epoch(
    model, dataset, epoch, output_dir, use_only_type, use_last_km=True
):
    dataset_path = output_dir + f"/gen__epoch_{epoch}"

    if os.path.exists(dataset_path):
        return dataset_path

    # Clean up temporary directory
    import gc
    import shutil
    import tempfile

    from dataset_augmenter import DatasetAugmenter

    from mttl.models.utils import model_loader_helper

    with tempfile.TemporaryDirectory() as temp_directory:
        device = model.device

        # we need to save the model in the temp directory, this moves it to CPU so that VLLM
        # doesn't complain, one alternative is to use 2 gpus, one for generation, one for training
        if use_last_km:
            model.merge_and_save_base_model(temp_directory, device="cpu")
            model_name_or_path = temp_directory
        else:
            model_name_or_path = model.config.base_model

        # Generate a set of prompts
        augmenter = DatasetAugmenter(
            model_name_or_path,
            block_size=2048,
            max_continuation_length=768,
            num_generations=4,
            generation_top_p=0.95,
            num_devices=1,
            model_type="local",
            do_filtering=False,
        )
        for task in use_only_type.split(","):
            augmenter.add_task(task)
        synth_dataset = augmenter.augment(dataset=dataset)

        # Force garbage collection
        del augmenter
        gc.collect()
        torch.cuda.empty_cache()
        model.to(device)

        synth_dataset = DatasetDict({"train": synth_dataset})
        synth_dataset.save_to_disk(dataset_path)

    return dataset_path


def get_text_dataset(training_args):
    args = copy.deepcopy(training_args)
    args.dataset = training_args.text_dataset
    args.dataset_type = "text_dataset"
    text_datamodule = get_datamodule(args)
    text_dataset = text_datamodule.train_dataset
    return text_dataset


@dataclass
class KMIterArguments(ExpertConfig):
    loss_function: str = "dcd"
    generate_using_last_km: bool = True
    generate_every_n_epochs: int = 1
    # set the following if you want to enable the NQA callback during training
    text_dataset: str = "sordonia/narrativeqa_sanitized"
    nqa_dataset: str = "sordonia/narrativeqa_sanitized"


def train_km(training_args: KMIterArguments):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    # save mttl args
    training_args.save_config(training_args.output_dir)

    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    model_config = ExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name or args.finetune_task_name,
        modifier_config=args.modifier_config,
    )

    model = ExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    ).to("cuda")

    data_args = copy.deepcopy(training_args)
    data_args.dataset = training_args.nqa_dataset
    evaluator = NQAZeroShotEvaluator(data_args, generation_kwargs={})

    if training_args.loss_function == "dcd":
        loss_function = dcd_loss
    else:
        raise ValueError(f"Loss function {training_args.loss_function} not supported")

    text_dataset = get_text_dataset(training_args)
    synth_dataset_path = create_synthetic_data_for_epoch(
        model, text_dataset, 0, training_args.output_dir, training_args.use_only_type
    )
    synth_datamodule = create_datamodule(training_args, synth_dataset_path)

    (optimizer, scheduler), trainable_param_names = get_optimizer_and_scheduler(
        model, training_args, num_train_examples=len(synth_datamodule.train_dataset)
    )
    # compute number of trainable parameters
    num_trainable_params = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad
    )
    logger.info(f"Number of trainable parameters: {num_trainable_params // 1e6:.2f}M")

    pbar = tqdm(
        total=len(synth_datamodule.train_dataloader())
        * training_args.num_train_epochs
        // args.gradient_accumulation_steps
    )

    global_step = 0
    best_val = float("inf")
    met_logger = SimpleLogger(training_args.output_dir)

    val_loss, rougeL = do_evaluation(synth_datamodule, model, loss_function, evaluator)
    met_logger.log_metrics({"val_loss": val_loss, "rougeL": rougeL}, step=global_step)

    for epoch in range(args.num_train_epochs):
        epoch_end = False

        iter_train = iter(synth_datamodule.train_dataloader())
        while not epoch_end:
            loss_accum = 0.0
            model.train()
            optimizer.zero_grad()

            for step in range(args.gradient_accumulation_steps):
                try:
                    batch = next(iter_train)
                except StopIteration:
                    epoch_end = True
                    break

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    batch = transfer_batch_to_device(batch, "cuda")

                loss = loss_function(model, batch)
                loss = loss / args.gradient_accumulation_steps
                loss_accum += loss.detach()
                loss.backward()

            if loss_accum:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()
                optimizer.step()
                torch.cuda.synchronize()
                pbar.update(1)

                lr = optimizer.param_groups[0]["lr"]
                met_logger.log_metrics(
                    {"train_loss": loss_accum.item(), "grad_norm": norm, "lr": lr},
                    step=global_step,
                )
                logger.info(
                    f"Epoch {epoch}, Loss: {loss_accum.item():.5f}, Grad Norm: {norm:.5f}, LR: {lr:.6f}"
                )

            global_step += 1

        val_loss, rougeL = do_evaluation(
            synth_datamodule, model, loss_function, evaluator
        )
        met_logger.log_metrics(
            {"val_loss": val_loss, "rougeL": rougeL}, step=global_step
        )

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(training_args.output_dir + "/best_model")
            logger.info(f"Saving model to {training_args.output_dir}")

        if (
            epoch % training_args.generate_every_n_epochs == 0
            and epoch != training_args.num_train_epochs - 1
        ):
            # regenerate synthetic data for next epoch!
            synth_dataset_path = create_synthetic_data_for_epoch(
                model,
                text_dataset,
                epoch + 1,
                training_args.output_dir,
                training_args.use_only_type,
                training_args.generate_using_last_km,
            )
            synth_datamodule = create_datamodule(training_args, synth_dataset_path)


if __name__ == "__main__":
    args = KMIterArguments.parse()
    train_km(args)
