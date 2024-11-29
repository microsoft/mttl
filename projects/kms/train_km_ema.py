import copy
import json
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning_fabric import seed_everything
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
from mttl.models.km_model import EMAExpertModel, EMAExpertModelConfig
from mttl.models.utils import transfer_batch_to_device
from mttl.utils import create_library, remote_login, seed_everything, upload_library

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule
from projects.kms.utils.nqa_datamodule import NQADatamodule  # noqa: F401
from projects.kms.utils.nqa_evaluator import NQAZeroShotEvaluator
from projects.kms.utils.quality_evaluator import QualityEvaluator
from projects.kms.utils.simple_utils import (
    SimpleLogger,
    do_evaluation,
    ema_dcd_loss,
    lm_loss,
)

torch.set_float32_matmul_precision("high")


@dataclass
class KMArguments(ExpertConfig):
    loss_function: str = "dcd"
    evaluate_on: str = "nqa"


def train_km(training_args: KMArguments):
    seed_everything(training_args.seed)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    if is_main_process():
        training_args.save_config(training_args.output_dir)

    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    model_config = EMAExpertModelConfig(
        base_model=args.model,
        modifier_config=args.modifier_config,
    )

    device = get_device()
    raw_model = model = EMAExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    ).to(device)

    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()])

    # Set default expert to KM
    raw_model.set_default_expert(raw_model.config.default_expert)

    # build evaluator
    data_args = copy.deepcopy(training_args)
    if training_args.evaluate_on == "nqa":
        eval_metric = "rougeL"
        evaluator = NQAZeroShotEvaluator(data_args)
    elif training_args.evaluate_on == "quality":
        eval_metric = "accuracy"
        evaluator = QualityEvaluator(data_args)

    if training_args.loss_function == "dcd":
        loss_function = ema_dcd_loss
    elif training_args.loss_function == "lm":
        loss_function = lm_loss
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

    # in `get_optimizer_and_scheduler`, we compute
    # args.total_steps = math.ceil(num_train_examples / global_bs) * args.num_train_epochs

    pbar = tqdm(
        total=training_args.total_steps,
        disable=not is_main_process(),
    )

    global_step = 0
    best_val = val_loss = float("inf")
    met_logger = SimpleLogger(training_args.output_dir)

    if training_args.eval_before_training:
        val_loss, eval_score = do_evaluation(
            datamodule, model, loss_function, evaluator
        )
        logger.info(f"Validation Loss: {val_loss}, {eval_metric}: {eval_score}")
        met_logger.log_metrics(
            {"val_loss": val_loss, eval_metric: eval_score}, step=global_step
        )

    # Handle "step" vs "epoch" logic for training and testing
    assert (
        training_args.total_steps > 0
    ), "`training_steps` should have been computed in `get_optimizer_and_scheduler`"
    assert (
        training_args.eval_every is None or training_args.eval_every > 0
    ), "`eval_every` should be None or > 0"

    epoch = 0
    finished = False
    iter_train = iter(datamodule.train_dataloader())

    while not finished:
        epoch_finished = False
        loss_accum = 0.0
        model.train()
        optimizer.zero_grad()

        for step in range(args.gradient_accumulation_steps):
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(datamodule.train_dataloader())
                epoch_finished = True
                epoch += 1

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

            # Do the EMA update
            model.ema_update()

            lr = optimizer.param_groups[0]["lr"]
            met_logger.log_metrics(
                {"train_loss": loss_accum.item(), "grad_norm": norm, "lr": lr},
                step=global_step,
            )

            logger.info(
                f"Epoch {epoch + 1},"
                f" Loss: {loss_accum:.4f},"
                f" Norm: {norm:.4f},"
                f" Lr: {scheduler.get_last_lr()[0]:.4f},"
                f" Val: {best_val:.4f} ({val_loss:.4f})"
            )

        global_step += 1

        do_eval_on_step = (
            training_args.eval_every and global_step % training_args.eval_every == 0
        )
        do_eval_on_epoch = (
            training_args.eval_every_n_epoch
            and epoch_finished
            and epoch % training_args.eval_every_n_epoch == 0
        )
        if do_eval_on_step or do_eval_on_epoch:
            val_loss, eval_score = do_evaluation(
                datamodule, model, loss_function, evaluator
            )
            logger.info(f"Validation Loss: {val_loss}, {eval_metric}: {eval_score}")
            met_logger.log_metrics(
                {"val_loss": val_loss, eval_metric: eval_score}, step=global_step
            )

            if val_loss < best_val and is_main_process():
                best_val = val_loss
                raw_model.save_pretrained(training_args.output_dir + "/best_model")
                training_args.save_config(training_args.output_dir + "/best_model")
                logger.info(f"Saving model to {training_args.output_dir}")

        if global_step >= training_args.total_steps:
            break

    # Also save last model
    raw_model.save_pretrained(training_args.output_dir + "/last_model")
    training_args.save_config(training_args.output_dir + "/last_model")


if __name__ == "__main__":
    args = KMArguments.parse(raise_error=False)
    args.trainable_param_names = ".*KM.*"
    train_km(args)
