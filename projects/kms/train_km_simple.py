import copy
import json
import logging
import os
import random
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning_fabric import seed_everything
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule
from projects.kms.utils.nqa_datamodule import NQADatamodule

# isort: split

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
from mttl.utils import create_library, remote_login, seed_everything, upload_library
from projects.kms.utils.nqa_evaluator import NQAZeroShotEvaluator
from projects.kms.utils.quality_evaluator import QualityEvaluator
from projects.kms.utils.simple_utils import (
    SimpleLogger,
    dcd_loss,
    do_evaluation,
    lm_loss,
    print_metrics,
    teacher_force_dcd_loss,
)
from projects.kms.utils.wiki_mmlu_evaluator import WikiMMLUEvaluator

torch.set_float32_matmul_precision("high")


train_datasets = {
    "quality-km-llama-8b": "az://mttldata/quality-summaries-qa-llama-8b-instruct",
    "quality-km-phi3-med": "az://mttldata/quality-summaries-qa-phi-3-medium",
    "wiki-km-llama-8b": "az://mttldata/wiki-top-20-summaries-qa-llama-8b-instruct",
    "wiki-km-phi3-med": "az://mttldata/wiki-top-20-summaries-qa-phi-3-medium",
    "nqa-km-phi3-med": "az://mttldata/nqa-summaries-qa-phi-3-medium",
    "nqa-km-llama-8b": "az://mttldata/nqa-summaries-qa-llama-8b-instruct",
}


evaluate_datasets = {
    "nqa": "az://mttldata/narrativeqa-sanitized",
    "wiki": "az://mttldata/wiki-top-20-sanitized",
    "quality": "az://mttldata/quality-sanitized",
    "wiki-rag": "az://mttldata/wiki-top-20-sanitized-rag",
}

evaluate_metrics = {
    "nqa": "rougeL",
    "wiki": "accuracy",
    "wiki-rag": "accuracy",
    "quality": "accuracy",
}

evaluate_class = {
    "nqa": NQAZeroShotEvaluator,
    "wiki": WikiMMLUEvaluator,
    "wiki-rag": WikiMMLUEvaluator,
    "quality": QualityEvaluator,
}


@dataclass
class KMArguments(ExpertConfig):
    loss_function: str = "dcd"
    evaluate_on: str = "nqa"
    logit_factor: float = 1.0
    hidden_factor: float = 1.0
    temp: float = 1.0
    loss_on_topk: float = None


def train_km(training_args: KMArguments):
    seed_everything(training_args.seed)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    if is_main_process():
        training_args.save_config(training_args.output_dir)

    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token, raise_error=False)

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

    # build evaluator
    data_args = copy.deepcopy(training_args)
    data_args.dataset = evaluate_datasets[training_args.evaluate_on]
    evaluator = evaluate_class[training_args.evaluate_on](data_args)
    eval_metric = evaluate_metrics[training_args.evaluate_on]

    datamodule = get_datamodule(training_args)
    (optimizer, scheduler), trainable_param_names = get_optimizer_and_scheduler(
        model, training_args, num_train_examples=len(datamodule.train_dataset)
    )

    if training_args.loss_function == "dcd":
        loss_function = partial(
            dcd_loss,
            logit_factor=training_args.logit_factor,
            hidden_factor=training_args.hidden_factor,
            loss_on_topk=training_args.loss_on_topk,
            tokenizer=datamodule.tokenizer,
            temp=training_args.temp,
        )
    elif training_args.loss_function == "teacher_force_dcd":
        loss_function = partial(
            teacher_force_dcd_loss,
            logit_factor=training_args.logit_factor,
            hidden_factor=training_args.hidden_factor,
            temp=training_args.temp,
        )
    elif training_args.loss_function == "lm":
        loss_function = lm_loss
    elif training_args.loss_function == "summary_lm":
        loss_function = partial(lm_loss, prefix="nc_")
    else:
        raise ValueError(f"Loss function {training_args.loss_function} not supported")

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
    eval_score_so_far = []
    met_logger = SimpleLogger(training_args.output_dir)

    if training_args.eval_before_training:
        val_loss, eval_score = do_evaluation(
            datamodule, model, loss_function, evaluator
        )
        met_logger.log_metrics(
            {"val_loss": val_loss, eval_metric: eval_score}, step=global_step
        )

        logger.info(f"Validation Loss: {val_loss}, {eval_metric}: {eval_score}")
        logger.info(
            f"Losses so far: {print_metrics(met_logger.get_metric('val_loss'))}"
        )
        logger.info(f"Eval so far: {print_metrics(met_logger.get_metric(eval_metric))}")

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
                p_training = global_step / training_args.total_steps
                loss = loss_function(model, batch)  # , p_training=p_training)

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

            met_logger.log_metrics(
                {"val_loss": val_loss, eval_metric: eval_score}, step=global_step
            )

            logger.info(f"Validation Loss: {val_loss}, {eval_metric}: {eval_score}")
            logger.info(
                f"Losses so far: {print_metrics(met_logger.get_metric('val_loss'))}"
            )
            logger.info(
                f"Eval so far: {print_metrics(met_logger.get_metric(eval_metric))}"
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
    args = KMArguments.parse()
    train_km(args)
