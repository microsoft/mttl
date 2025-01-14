import copy
import json
import logging
import os
from dataclasses import dataclass
from functools import partial

import torch
from lightning_fabric import seed_everything
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule

# isort: split

from mttl.arguments import MultiExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.logging import logger, setup_logging
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device
from projects.kms.train_km_simple import (
    evaluate_class,
    evaluate_datasets,
    evaluate_metrics,
)
from projects.kms.utils.simple_utils import (
    SimpleLogger,
    do_evaluation,
    lm_loss,
    print_metrics,
)

# isort: split

# import Selector before args
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.km_model import KEMoEModel, KEMoEModelConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import remote_login
from projects.kms.train_km_simple import KMArguments

train_datasets = {
    "nqa": "az://mttldata/narrativeqa-sanitized",
    "nqa-rag": "pclucas14/nqa-RAG-64",
    "quality": "az://mttldata/quality-sanitized",
    "quality-rag": "pclucas14/quality-RAG-64",
}


@dataclass
class KEArguments(MultiExpertConfig, KMArguments):
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = None
    # Where to save the KE expert
    ke_hf_path: str = None
    # Whether to retrieve passages
    use_rag: bool = False
    # offload experts
    offload_experts: bool = False
    # max kms
    max_kms: int = None
    # split file for subsampling
    subsample_file: str = None

    def __post_init__(self):
        super().__post_init__()

        # Automating some args
        if self.use_rag:
            if not self.evaluate_on.endswith("-rag"):
                logger.warning(f"Overwriting `evaluate_on` to {self.evaluate_on}-rag")
                self.evaluate_on += "-rag"
            self.include_context = True

        self.dataset = train_datasets[self.evaluate_on]
        eval_on = self.evaluate_on.split("-")[0]
        dataset_type = {"nqa": "narrativeqa", "quality": "quality"}[eval_on]
        if self.dataset_type != dataset_type:
            logger.warning(f"Overwriting `dataset_type` to {dataset_type}")
            self.dataset_type = dataset_type

        if self.finetune_task_name is None:
            self.finetune_task_name = {
                "quality": "splits/quality/quality_full.json",
                "nqa": "splits/nqa/nqa_full.json",
            }[eval_on]
            logger.warning(
                f"Overwriting `finetune_task_name` to {self.finetune_task_name}"
            )

        # Allow to set trainable tasks from a json split file (e.g. nqa_mini_split.json)
        if isinstance(
            self.finetune_task_name, str
        ) and self.finetune_task_name.endswith(".json"):
            if self.finetune_task_name != self.subsample_file:
                logger.warning(
                    f"Overwriting `subsample_file` to {self.finetune_task_name}"
                )
                self.subsample_file = self.finetune_task_name

            with open(self.finetune_task_name, "r") as f:
                split_dict = json.load(f)

            # HACK so that QAEvalArguments can still work
            if hasattr(self, "split"):
                tasks = split_dict[self.split]
            else:
                tasks = split_dict["train"] + split_dict["dev"]
            logger.info(f"Setting finetune_task_name to {tasks}")
            self.finetune_task_name = tasks

        if self.max_kms is not None:
            import random

            random.shuffle(self.finetune_task_name)
            logger.info(f"Selecting {self.max_kms} tasks to finetune on")
            self.finetune_task_name = self.finetune_task_name[: self.max_kms]


def train_ke(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    if is_main_process():
        training_args.save_config(training_args.output_dir)

    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    if training_args.library_id:
        logger.info("Loading expert library: %s", training_args.library_id)

        model_config = KEMoEModelConfig(
            base_model=training_args.model,
            library_id=training_args.library_id,
            expert_selection=args.finetune_task_name,
            selector_config=training_args.selector_config,
            offload_experts=training_args.offload_experts,
        )
        model = KEMoEModel(model_config)

        if model.ke_expert_name not in training_args.trainable_param_names:
            # Let's provide a fix that works for the current setup
            logger.warning("Overwriting `trainable_param_names` to include the KE")
            training_args.trainable_param_names = f".*{model.ke_expert_name}.*"

        # for which we have trained KM experts
        if not training_args.finetune_task_name:
            logger.info(
                f"Setting `finetune_task_name` to match with experts in the model"
            )
            training_args.finetune_task_name = list(
                filter(lambda x: x != model.ke_expert_name, model.experts_names)
            )
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

    device = get_device()
    raw_model = model = model.to(device)

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

    # For KE training, loss function is always LM
    loss_function = lm_loss

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
    best_val = val_loss = float("inf")
    eval_score_so_far = []
    met_logger = SimpleLogger(training_args.output_dir)

    if training_args.eval_before_training:
        val_loss, eval_score = do_evaluation(
            datamodule,
            model,
            loss_function,
            (
                evaluator
                if training_args.callback_during_training
                and not training_args.offload_experts
                else None
            ),
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
                (
                    evaluator
                    if training_args.callback_during_training
                    and not training_args.offload_experts
                    else None
                ),
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

    # Can we load the best model and evaluate it ?
    model_class = type(model)
    del model
    model = model_class.from_pretrained(training_args.output_dir + "/best_model")

    # Maybe save to Expert Library
    if args.ke_hf_path:
        # TODO: make sure that pushing expert in MoE works
        if isinstance(model, KEMoEModel):
            ke_expert = model.get_expert_instance(model.ke_expert_name)
        else:
            ke_expert = model.as_expert()

        # create a library and upload that expert
        lib_path, exp_name = args.ke_hf_path.rsplit("/", 1)
        expert_library = ExpertLibrary.get_expert_library(lib_path, create=True)
        expert_library.add_expert(ke_expert, exp_name, force=True)


if __name__ == "__main__":
    args = KEArguments.parse()
    assert args.dataset_config

    if os.path.exists(args.output_dir + "/last_model/mttl_weights.bin"):
        logger.info("Model already trained, skipping")
        exit(0)

    train_ke(args)
