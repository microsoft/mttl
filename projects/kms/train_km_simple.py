import copy
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from nqa_datamodule import NQADatamodule  # noqa: F401
from tqdm import tqdm
from utils.callbacks import LogMttlArgs

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig, disable_modifiers
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device
from mttl.utils import create_library, remote_login, upload_library


class SimpleLogger:
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "metrics.json")
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def log_metrics(self, metrics, step=None):
        lines = []

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            lines.append({"name": k, "value": v, "step": step})

        try:
            with open(self.output_file, "a+") as f:
                for l in lines:
                    f.write(json.dumps(l) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")


def dcd_loss(model, inputs):
    kl_loss = torch.nn.KLDivLoss(reduction="none")

    # document + small task prompt + task output (e.g. summary, or question and answer)
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]

    # small task prompt + task output (e.g. summary, or question and answer)
    nc_input_ids = inputs["nc_input_ids"]
    nc_labels = inputs["nc_labels"]
    nc_attention_mask = inputs["nc_attention_mask"]
    position_ids = None

    # for the context-aware pass, we need to disable the adapter
    with disable_modifiers(model):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            target_hidden_states = [
                hidden_state[labels != -100, ...]
                for hidden_state in outputs.hidden_states
            ]
            target_logits = outputs.logits[labels != -100, :]

    outputs = model(
        input_ids=nc_input_ids,
        attention_mask=nc_attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    losses = []
    for actual_states, target_states in zip(
        outputs.hidden_states, target_hidden_states
    ):
        actual_states = actual_states[nc_labels != -100, :]

        # Calculate mean magnitude of target states
        mean = torch.sum(torch.abs(target_states)) / (actual_states.size(-1))
        if actual_states.size(0) != target_states.size(0):
            # this shouldn't happen, but sometimes it does probably due to weird tokenization issues
            logger.warning("Skipping batch due to mismatch in shape")
            continue

        losses.append(
            # Loss is the mean abs difference between target and predicted states,
            # normalised by mean magnitude of target states
            torch.sum(torch.abs(actual_states - target_states))
            / (mean * np.prod(target_states.shape))
        )

    if len(losses) == 0:
        # this happens when shape mismatch due to tokenization issues, should happen rarely
        fake_loss = actual_states.sum() * 0.0
        return fake_loss

    loss = torch.mean(torch.stack(losses))

    # Add KL divergence between target and predicted output distributions to loss
    target_probs = F.softmax(target_logits, dim=-1)
    preds = F.log_softmax(outputs.logits[nc_labels != -100, ...], dim=-1)
    kl_loss = kl_loss(preds, target_probs).sum(dim=-1).mean()
    loss = loss + kl_loss
    return loss


def do_evaluation(datamodule, model, loss_function, nqa_evaluator) -> bool:
    # validation
    for batch in tqdm(datamodule.val_dataloader()):
        val_loss = 0.0
        step = 0.0

        with torch.no_grad():
            batch = transfer_batch_to_device(batch, "cuda")
            val_loss += loss_function(model, batch).item()
            step += 1
        val_loss /= step

    rougeL = nqa_evaluator.evaluate(model, "dev")
    logger.info(f"Validation Loss: {val_loss}, ROUGE-L: {rougeL}")
    return val_loss


@dataclass
class KMArguments(ExpertConfig):
    loss_function: str = "dcd"
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = "sordonia/narrativeqa"


def train_km(training_args: KMArguments):
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

    if args.library_id:
        expert_library = create_library(args)

        if model_config.expert_name in expert_library:
            # gracefully exit if expert already exists in library
            logger.warning(
                f"Expert {model_config.expert_name} already exists in library!"
            )
            return

    model = ExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    ).to("cuda")

    # load the NQA callback to monitor zero-shot performance
    from nqa_evaluator import NQAZeroShotEvaluator

    data_args = copy.deepcopy(training_args)
    data_args.dataset = training_args.nqa_dataset
    evaluator = NQAZeroShotEvaluator(data_args, generation_kwargs={})

    loss_function = dcd_loss if training_args.loss_function == "dcd" else None

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
                    batch = transfer_batch_to_device(batch, "cuda")

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

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(training_args.output_dir)
            logger.info(f"Saving model to {training_args.output}")


if __name__ == "__main__":
    args = KMArguments.parse()
    train_km(args)
