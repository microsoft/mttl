import copy
import os
import random
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule
from projects.kms.utils.nqa_datamodule import NQADatamodule
from projects.kms.utils.nqa_evaluator import NQAZeroShotEvaluator
from projects.kms.utils.pit_datamodule import PITDatasetModule

# isort: split

from mttl.arguments import MultiExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
    seed_everything,
)
from mttl.logging import logger, setup_logging
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.library.expert import load_expert
from mttl.models.utils import transfer_batch_to_device
from projects.kms.train_km_simple import (
    evaluate_class,
    evaluate_datasets,
    evaluate_metrics,
)
from projects.kms.utils.quality_datamodule import QualityDatamodule
from projects.kms.utils.quality_evaluator import QualityEvaluator
from projects.kms.utils.simple_utils import (
    EarlyStopper,
    SimpleLogger,
    do_evaluation,
    lm_loss,
    mc_loss,
    print_metrics,
)

# isort: split

# import Selector before args
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import remote_login
from projects.kms.train_km_simple import KMArguments
from projects.kms.utils.km_model import KEMoEModel, KEMoEModelConfig


@dataclass
class KEArguments(MultiExpertConfig, KMArguments):
    # Where to save the KE expert
    ke_uri: str = None
    # max kms
    max_kms: int = None
    # whether to overwrite / force upload of the new expert
    force: bool = False
    # keep on cpu
    cpu_offload: bool = False


def train_ke(training_args):
    seed_everything(training_args.seed)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    if is_main_process():
        training_args.save_config(training_args.output_dir)

    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    # build evaluator
    if training_args.dataset_type == "quality":
        eval_metric = "accuracy"
        evaluator = QualityEvaluator(training_args)
    elif training_args.dataset_type == "narrativeqa":
        eval_metric = "rougeL"
        evaluator = NQAZeroShotEvaluator(training_args)

    datamodule = get_datamodule(training_args)

    if training_args.library_id:
        logger.info("Loading expert library: %s", training_args.library_id)

        expert_selection = datamodule.train_task_names
        if training_args.max_kms:
            random.shuffle(expert_selection)
            expert_selection = expert_selection[: training_args.max_kms]

        expert_selection += datamodule.dev_task_names

        model_config = KEMoEModelConfig(
            base_model=training_args.model,
            library_id=training_args.library_id,
            expert_selection=expert_selection,
            selector_config=training_args.selector_config,
            cpu_offload=training_args.cpu_offload,
        )
        model = KEMoEModel(
            model_config,
            attn_implementation=training_args.attn_implementation
        )

        if model.ke_expert_name not in training_args.trainable_param_names:
            # Let's provide a fix that works for the current setup
            logger.warning("Overwriting `trainable_param_names` to include the KE")
            training_args.trainable_param_names = f".*{model.ke_expert_name}.*"

    else:
        logger.info("Loading model without expert library")
        model_config = ExpertModelConfig(
            base_model=args.model,
            expert_name=args.expert_name or "KE",
            modifier_config=args.modifier_config,
        )

        model = ExpertModel(
            model_config,
            load_in_4bit=training_args.load_in_4bit,
            load_in_8bit=training_args.load_in_8bit,
            device_map=training_args.device_map,
            attn_implementation=training_args.attn_implementation,
        )

    # deactivate use_cache for Phi
    if "Phi" in args.model:
        model.model.config.use_cache = False

    device = get_device()
    raw_model = model = model.to(device)

    if is_dist_avail_and_initialized():
        model.model = DDP(model.model, device_ids=[get_local_rank()])

    (optimizer, scheduler), _ = get_optimizer_and_scheduler(
        model, training_args, num_train_examples=len(datamodule.train_dataset)
    )

    # For KE training, loss function is always LM
    is_quality = isinstance(datamodule, QualityDatamodule)
    loss_function = mc_loss if is_quality else lm_loss

    # compute number of trainable parameters
    num_trainable_params = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad
    )
    logger.info(f"Number of trainable parameters: {num_trainable_params // 1e6:.2f}M")

    pbar = tqdm(
        total=training_args.total_steps,
        disable=not is_main_process(),
    )

    global_step = 0
    best_score = eval_score = 0
    best_val = val_loss = float("inf")
    met_logger = SimpleLogger(training_args.output_dir)

    # early stopper
    early_stopper = None
    if training_args.patience is not None:
        early_stopper = EarlyStopper(patience=training_args.patience, mode="min")

    if training_args.eval_before_training:
        val_loss, eval_score = do_evaluation(
            datamodule,
            model,
            loss_function,
            evaluator,
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
                loss = loss_function(model, batch)

            model.require_backward_grad_sync = (
                step + 1 == args.gradient_accumulation_steps
            )
            loss = loss / args.gradient_accumulation_steps
            loss_accum += loss.detach()
            loss.backward()
            del loss, batch
            torch.cuda.empty_cache()

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
            training_args.eval_every
            and training_args.eval_every > 0
            and global_step % training_args.eval_every == 0
        )
        do_eval_on_epoch = (
            training_args.eval_every_n_epoch
            and training_args.eval_every_n_epoch > 0
            and epoch_finished
            and epoch % training_args.eval_every_n_epoch == 0
        )
        if do_eval_on_step or do_eval_on_epoch:
            val_loss, eval_score = do_evaluation(
                datamodule,
                model,
                loss_function,
                (evaluator if training_args.callback_during_training else None),
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
                best_score = eval_score
                model.save_pretrained(training_args.output_dir + "/best_model")
                training_args.save_config(training_args.output_dir + "/best_model")
                logger.info(f"Saving model to {training_args.output_dir}")

            if early_stopper and early_stopper(val_loss):
                logger.info(f"Early stopping after {global_step} steps")
                break

        if global_step >= training_args.total_steps:
            break

    # reload the best model
    raw_model.load_weights(training_args.output_dir + "/best_model")

    val_loss, eval_score = do_evaluation(datamodule, model, loss_function, evaluator)
    logger.info(f"Final Validation Loss: {val_loss}, {eval_metric}: {eval_score}")
    met_logger.log_metrics(
        {"best_val_loss": val_loss, f"best_{eval_metric}": eval_score}, step=global_step
    )

    # Also save last model
    if is_main_process():
        raw_model.save_pretrained(training_args.output_dir + "/last_model")
        training_args.save_config(training_args.output_dir + "/last_model")

    # Maybe save to Expert Library
    if args.ke_uri and is_main_process():
        if isinstance(model, KEMoEModel):
            ke_expert = model.get_expert_instance(model.ke_expert_name)
        else:
            ke_expert = model.as_expert()

        # create a library and upload that expert
        lib_path, exp_name = args.ke_uri.rsplit("/", 1)
        expert_library = ExpertLibrary.get_expert_library(lib_path, create=True)
        expert_library.add_expert(ke_expert, exp_name, force=True)


if __name__ == "__main__":
    args = KEArguments.parse()
    assert args.dataset_config

    # check if KE expert already exists
    if not args.force and os.path.exists(
        args.output_dir + "/last_model/mttl_weights.bin"
    ):
        logger.warning(f"Found expert checkpoint. in {args.output_dir}/last_model")
        exit(0)
    elif not args.force and args.ke_uri:
        try:
            expert = load_expert(args.ke_uri)
            logger.warning(f"Found expert in {args.ke_uri}")
            exit(0)
        except:
            pass

    train_ke(args)
