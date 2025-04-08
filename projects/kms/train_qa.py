import os
import random
from dataclasses import dataclass
from functools import partial

import torch
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
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    seed_everything,
)
from mttl.logging import logger, setup_logging
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device
from projects.kms.utils.longhealth_datamodule import LonghealthDatamodule
from projects.kms.utils.longhealth_evaluator import LonghealthEvaluator
from projects.kms.utils.quality_datamodule import QualityDatamodule
from projects.kms.utils.quality_evaluator import GenQualityEvaluator, QualityEvaluator
from projects.kms.utils.simple_utils import (
    EarlyStopper,
    SimpleLogger,
    cpu_offload,
    do_evaluation,
    lm_loss,
    mc_loss,
    mc_loss_iterative,
    print_metrics,
)

# isort: split

# import Selector before args
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import get_ram, get_vram, remote_login
from projects.kms.train_km_simple import KMArguments
from projects.kms.utils.km_model import KEMoEModel, KEMoEModelConfig


@dataclass
class KEArguments(MultiExpertConfig, KMArguments):
    # Where to save the KE expert
    ke_uri: str = None
    # Reload KE expert to do fine-tuning
    ke_path: str = None
    #
    max_train_tasks: int = -1
    # whether to overwrite / force upload of the new expert
    force: bool = False
    # keep on cpu
    cpu_offload: bool = False
    # if false, we train & eval
    do_eval: bool = False
    # whether evaluator should be verbose
    verbose: bool = False


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
        split = "dev"
    elif training_args.dataset_type == "narrativeqa":
        eval_metric = "rougeL"
        evaluator = NQAZeroShotEvaluator(training_args)
        split = "test"
    elif training_args.dataset_type == "longhealth":
        eval_metric = "accuracy"
        evaluator = LonghealthEvaluator(training_args)
        split = "test"

    datamodule = get_datamodule(training_args)
    expert_selection = datamodule.train_task_names

    if training_args.max_train_tasks > 0:
        random.shuffle(expert_selection)
        expert_selection = expert_selection[: training_args.max_train_tasks]
    if split == "dev":
        expert_selection += datamodule.dev_task_names
        eval_task_names = datamodule.dev_task_names
    elif split == "test":
        expert_selection += datamodule.test_task_names
        eval_task_names = datamodule.test_task_names

    # TODO max eval tasks  ?

    if training_args.do_eval:
        expert_selection = eval_task_names
    else:
        expert_selection = list(set(expert_selection))
    logger.info(f"Tasks selected: {len(expert_selection)}")

    training_args.finetune_task_name = ",".join(expert_selection)
    datamodule = get_datamodule(training_args)

    if training_args.library_id:
        logger.info("Loading expert library: %s", training_args.library_id)

        model_config = KEMoEModelConfig(
            base_model=training_args.model,
            library_id=training_args.library_id,
            expert_selection=expert_selection,
            selector_config=training_args.selector_config,
            eval_cpu_offload=training_args.cpu_offload,
        )
        model = KEMoEModel(
            model_config,
            attn_implementation=training_args.attn_implementation,
            precision=training_args.precision,
            device_map=get_device() if not training_args.cpu_offload else "cpu",
        )

        if training_args.cpu_offload:
            for n, p in model.named_parameters():
                if "lora" not in n or "KE" in n:
                    p.data = p.data.to(get_device())
            for b in model.buffers():
                b.data = b.data.to(get_device())

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
            device_map=get_device(),
            precision=training_args.precision,
            attn_implementation=training_args.attn_implementation,
        )

    # deactivate use_cache for Phi
    if "Phi" in args.model:
        model.model.config.use_cache = False

    is_quality = isinstance(datamodule, QualityDatamodule)
    if is_quality and args.cpu_offload:
        loss_function = partial(
            mc_loss_iterative, pad_token_id=datamodule.tokenizer.pad_token_id
        )
    elif is_quality:
        loss_function = mc_loss
    else:
        loss_function = lm_loss
    device = get_device()

    if not training_args.do_eval:
        (optimizer, scheduler), _ = get_optimizer_and_scheduler(
            model, training_args, num_train_examples=len(datamodule.train_dataset)
        )

        # compute number of trainable parameters
        num_trainable_params = sum(
            p.numel() for name, p in model.named_parameters() if p.requires_grad
        )
        logger.info(
            f"Number of trainable parameters: {num_trainable_params // 1e6:.2f}M"
        )

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
                evaluator=evaluator,
                evaluator_split=split,
                split=split,
                verbose=training_args.verbose,
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
                    batch = next(iter_train)
                    epoch_finished = True
                    epoch += 1

                with cpu_offload(
                    model, batch["task_names"], enable=training_args.cpu_offload
                ):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    ):
                        batch = transfer_batch_to_device(batch, device)
                        loss = loss_function(model, batch)

                    loss = loss / args.gradient_accumulation_steps
                    loss_accum += loss.detach()
                    loss.backward()

                del loss, batch
                torch.cuda.empty_cache()

            if loss_accum:
                # Sum gradients across all workers
                if is_dist_avail_and_initialized():
                    torch.distributed.barrier()

                    for p in model.parameters():
                        if p.grad is not None:
                            torch.distributed.all_reduce(
                                p.grad, op=torch.distributed.ReduceOp.SUM
                            )
                            p.grad /= get_world_size()

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
                    f" Val: {best_val:.4f} ({val_loss:.4f}),"
                    f" {get_ram()},"
                    f" {get_vram()}"
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
                    evaluator_split=split,
                    split=split,
                    verbose=training_args.verbose,
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
    """Reload the weights from a saved directory."""
    if not os.path.exists(training_args.output_dir + "/best_model"):
        raise ValueError("Couldn't find a best model directory!")

    model.load_weights(training_args.output_dir + "/best_model")

    if is_main_process():
        os.makedirs(training_args.output_dir + "/eval_output/", exist_ok=True)

    val_loss, eval_score = do_evaluation(
        datamodule,
        model,
        loss_function,
        evaluator,
        evaluator_split=split,
        split=split,
        output_path=training_args.output_dir + "/eval_output/",
        verbose=training_args.verbose,
    )

    logger.info(f"Final Validation Loss: {val_loss}, {eval_metric}: {eval_score}")
    with open(f"{training_args.output_dir}/final_eval.json", "w") as f:
        import json

        f.write(json.dumps({"best_val_loss": val_loss, eval_metric: eval_score}))

    if not training_args.do_eval:
        met_logger.log_metrics(
            {"best_val_loss": val_loss, f"best_{eval_metric}": eval_score},
            step=global_step,
        )

        # Also save last model
        if is_main_process():
            model.save_pretrained(training_args.output_dir + "/last_model")
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

    train_ke(args)
