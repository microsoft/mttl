import json
import os

import torch
import torch.distributed as dist
from torch.nn import functional as F
from tqdm import tqdm

from mttl.dist_utils import (
    distributed_mean,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.logging import logger
from mttl.models.expert_model import disable_modifiers
from mttl.models.utils import transfer_batch_to_device


def dcd_loss(model, inputs, logit_factor=1.0, hidden_factor=1.0):
    """Deep Contextual Distillation loss."""
    kl_loss = torch.nn.KLDivLoss(reduction="none")

    # document + small task prompt + task output (e.g. summary, or question and answer)
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]

    # small task prompt + task output (e.g. summary, or question and answer)
    nc_input_ids = inputs["nc_input_ids"]
    nc_labels = inputs["nc_labels"]
    nc_attention_mask = inputs["nc_attention_mask"]

    # length of the context!
    all_length = attention_mask.sum(1)
    context_length = all_length - nc_attention_mask.sum(1)
    position_ids = torch.arange(
        0,
        nc_input_ids.size(1),
        device=input_ids.device,
        dtype=torch.long,
    )
    position_ids = context_length.unsqueeze(1) + position_ids.unsqueeze(0)
    raw_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )

    # for the context-aware pass, we need to disable the adapter
    with disable_modifiers(raw_model):
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

    loss = 0.0
    losses = []

    if hidden_factor > 0:
        for layer_id, (actual_states, target_states) in enumerate(
            zip(outputs.hidden_states, target_hidden_states)
        ):
            actual_states = actual_states[nc_labels != -100, :]

            if actual_states.size(0) != target_states.size(0):
                # this shouldn't happen, but sometimes it does probably due to weird tokenization issues
                logger.warning("Skipping batch due to mismatch in shape")
                continue

            # Loss is the mean abs difference between target and predicted states,
            # normalised by mean magnitude of target states
            loss = (actual_states - target_states).abs().mean()
            loss = loss / target_states.abs().mean()
            losses.append(loss)

        # Can we log the `kl_loss` of different layers ?
        if len(losses) == 0:
            # this happens when shape mismatch due to tokenization issues, should happen rarely
            fake_loss = actual_states.sum() * 0.0
            return fake_loss

        loss = torch.mean(torch.stack(losses)) * hidden_factor

    # Add KL divergence between target and predicted output distributions to loss
    target_probs = F.softmax(target_logits, dim=-1)
    preds = F.log_softmax(outputs.logits[nc_labels != -100, ...], dim=-1)
    kl_loss = kl_loss(preds, target_probs).sum(dim=-1).mean()

    loss = loss + logit_factor * kl_loss
    return loss


def lm_loss(model, inputs):
    """Next-token prediction loss."""
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]

    assert input_ids.size() == labels.size()
    # assert that labels is either -100 or the same as input_ids
    assert torch.all((labels == -100) | (labels == input_ids))

    # NOTE: when using `LMTrainer` for training the KM, batch has unwanted keys.
    # This Trainer is also used for training the KE.
    # Also: I need **batch for the KE training with knowledge modules (the task names)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        task_names=inputs.get("task_names"),
    )
    return outputs.loss


def do_evaluation(datamodule, model, loss_function, evaluator) -> bool:
    val_loss = []
    for batch in tqdm(datamodule.val_dataloader(), disable=not is_main_process()):
        with torch.no_grad():
            batch = transfer_batch_to_device(batch, model.device)
            val_loss.append(loss_function(model, batch).item())

    val_loss = distributed_mean(val_loss, model.device)
    rougeL = evaluator.evaluate(model, "dev")
    logger.info(f"Validation Loss: {val_loss}, ROUGE-L: {rougeL}")
    return val_loss, rougeL


class SimpleLogger:
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "metrics.json")
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def log_metrics(self, metrics, step=None):
        from mttl.dist_utils import is_main_process

        if not is_main_process():
            return

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
