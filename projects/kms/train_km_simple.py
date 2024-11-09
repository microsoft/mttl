import copy
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from nqa_datamodule import NQADatamodule  # noqa: F401
from simple_utils import SimpleLogger, dcd_loss, do_evaluation
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig, disable_modifiers
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device
from mttl.utils import create_library, remote_login, upload_library

torch.set_float32_matmul_precision("high")


@dataclass
class KMArguments(ExpertConfig):
    loss_function: str = "dcd"
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = "sordonia/narrativeqa_sanitized"


def train_km(training_args: KMArguments):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    if is_main_process():
        training_args.save_config(training_args.output_dir)

    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    model_config = ExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name or args.finetune_task_name,
        modifier_config=args.modifier_config,
    )

    device = get_device()
    raw_model = model = ExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    ).to(device)

    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()])

    # load the NQA callback to monitor zero-shot performance
    from nqa_evaluator import NQAZeroShotEvaluator

    data_args = copy.deepcopy(training_args)
    data_args.dataset = training_args.nqa_dataset
    evaluator = NQAZeroShotEvaluator(data_args, generation_kwargs={})

    if training_args.loss_function == "dcd":
        loss_function = dcd_loss
    else:
        raise ValueError(f"Loss function {training_args.loss_function} not supported")

    datamodule = get_datamodule(training_args)
    (optimizer, scheduler), trainable_param_names = get_optimizer_and_scheduler(
        model, training_args, num_train_examples=len(datamodule.train_dataset)
    )
    # compute number of trainable parameters
    num_trainable_params = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad
    )
    logger.info(f"Number of trainable parameters: {num_trainable_params // 1e6:.2f}M")

    pbar = tqdm(
        total=len(datamodule.train_dataloader())
        * training_args.num_train_epochs
        // args.gradient_accumulation_steps
    )

    global_step = 0
    best_val = float("inf")
    met_logger = SimpleLogger(training_args.output_dir)

    val_loss, rougeL = do_evaluation(datamodule, model, loss_function, evaluator)
    met_logger.log_metrics({"val_loss": val_loss, "rougeL": rougeL}, step=global_step)

    for epoch in range(args.num_train_epochs):
        epoch_end = False

        iter_train = iter(datamodule.train_dataloader())
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
                    batch = transfer_batch_to_device(batch, device)

                loss = loss_function(model, batch)
                loss = loss / args.gradient_accumulation_steps
                loss_accum += loss.detach()
                loss.backward()

            if loss_accum:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()
                optimizer.step()
                torch.cuda.synchronize()  # wait for the GPU to finish work
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

        val_loss, rougeL = do_evaluation(datamodule, model, loss_function, evaluator)
        met_logger.log_metrics(
            {"val_loss": val_loss, "rougeL": rougeL}, step=global_step
        )

        if val_loss < best_val and is_main_process():
            best_val = val_loss
            raw_model.save_pretrained(training_args.output_dir)
            logger.info(f"Saving model to {training_args.output_dir}")


if __name__ == "__main__":
    args = KMArguments.parse()
    train_km(args)
