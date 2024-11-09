import json
import os

import torch
from torch.nn import functional as F
from tqdm import tqdm

from mttl.logging import logger
from mttl.models.expert_model import disable_modifiers
from mttl.models.utils import transfer_batch_to_device


def dcd_loss(model, inputs, logit_factor=1.0, hidden_factor=1.0):
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


def do_evaluation(datamodule, model, loss_function, evaluator) -> bool:
    # validation
    for batch in tqdm(datamodule.val_dataloader()):
        val_loss = 0.0
        step = 0.0

        with torch.no_grad():
            batch = transfer_batch_to_device(batch, "cuda")
            val_loss += loss_function(model, batch).item()
            step += 1
        val_loss /= step

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
