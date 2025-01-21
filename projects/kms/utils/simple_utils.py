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


def print_metrics(data):
    try:
        import numpy as np

        min_value = min(data)
        max_value = max(data)
        spark_chars = "▁▂▃▄▅▆▇█"

        # Handle the case where all data points are the same
        if max_value - min_value == 0:
            return spark_chars[3] * len(data)

        std = np.std(data)
        if std == 0:
            return "<std = 0>, skipping"

        min_value = min(data) - std
        max_value = max(data) + std

        # Scale data points to indices of spark_chars
        scaled_data = [
            int((value - min_value) / (max_value - min_value) * (len(spark_chars) - 1))
            for value in data
        ]

        # Map scaled data to corresponding sparkline characters
        return "".join(spark_chars[idx] for idx in scaled_data)
    except:
        return "<error>"


def dcd_loss(
    model,
    inputs,
    logit_factor=1.0,
    hidden_factor=1.0,
    loss_on_topk=None,
    tokenizer=None,
    temp=1.0,
):
    """Deep Contextual Distillation loss."""
    kl_loss = torch.nn.KLDivLoss(reduction="none")

    # document + small task prompt + task output (e.g. summary, or question and answer)
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    valid_idx = labels != -100

    # small task prompt + task output (e.g. summary, or question and answer)
    nc_input_ids = inputs["nc_input_ids"]
    nc_labels = inputs["nc_labels"]
    nc_attention_mask = inputs["nc_attention_mask"]
    nc_valid_idx = nc_labels != -100

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
                hidden_state[valid_idx] for hidden_state in outputs.hidden_states
            ]
            target_logits = outputs.logits[valid_idx]

    nc_outputs = model(
        input_ids=nc_input_ids,
        attention_mask=nc_attention_mask,
        output_hidden_states=True,
        return_dict=True,
        task_names=inputs.get("task_names"),  # Here
    )

    loss = 0.0
    losses = []

    if loss_on_topk is not None:
        with torch.no_grad():
            # per token losses for student and teacher

            # for the teacher
            bs, sq, D = outputs.logits.size()
            target_loss = F.cross_entropy(
                outputs.logits[:, :-1].flatten(0, 1),
                labels[:, 1:].flatten(0),
                reduction="none",
            )
            all_target_loss = target_loss.reshape(bs, sq - 1)
            target_loss = all_target_loss[valid_idx[:, :-1]]

            # for the student
            bs, sq, D = nc_outputs.logits.size()
            student_loss = F.cross_entropy(
                nc_outputs.logits[:, :-1].flatten(0, 1),
                nc_labels[:, 1:].flatten(0),
                reduction="none",
            )
            all_student_loss = student_loss.reshape(bs, sq - 1)
            student_loss = all_student_loss[nc_valid_idx[:, :-1]]

            score = all_target_loss.clone() * -1
            # add student loss
            score[valid_idx[:, :-1]] += student_loss
            score[~valid_idx[:, :-1]] = -1e9

            # hard token idx : bs, topk
            avg_valid_tokens = valid_idx[:, :-1].sum(1).float().mean().int().item()
            _, hard_token_idx = score.topk(
                k=int(avg_valid_tokens * loss_on_topk), dim=1
            )

            """
            # --- Debugging to make sure masking seems ok 
            mask = torch.zeros_like(score).bool()
            mask.scatter_(1, hard_token_idx, True)

            for bs_idx in range(bs):
                tea_ids = torch.where(valid_idx[bs_idx, :-1])[0]
                stu_ids = torch.where(nc_valid_idx[bs_idx, :-1])[0]
                hard_token_idx_ = torch.where(mask[bs_idx][valid_idx[bs_idx, :-1]])[0]

                for i in range(tea_ids.size(0)):
                    assert nc_input_ids[bs_idx, stu_ids[i]] == input_ids[bs_idx, tea_ids[i]]
                    token = tokenizer.decode(input_ids[bs_idx, tea_ids[i]].item())
                    tea_loss = all_target_loss[bs_idx, tea_ids[i]].item()
                    stu_loss = all_student_loss[bs_idx, stu_ids[i]].item()
                    hard_token = (i == hard_token_idx_).any().item()
                    fill = lambda x : (x + ' ' * (20 - len(x)))
                    if hard_token:
                        print(f'{fill(token)} | tea loss : {tea_loss:.2f} | stu loss : {stu_loss:.2f} | gap : {stu_loss - tea_loss:.2f} | hard token')
                    else:
                        print(f'{fill(token)} | tea loss : {tea_loss:.2f} | stu loss : {stu_loss:.2f} | gap : {stu_loss - tea_loss:.2f}')

            # --- Debug end 
            """

            # build a bool mask the same size of `score` that's True if the token is among the hard token
            # mask[i,j] = True if `j` in hard_token_idx[i]
            mask = torch.zeros_like(score).bool()
            mask.scatter_(1, hard_token_idx, True)
            hard_token_idx = mask[valid_idx[:, :-1]]

            # finally, let's extract 1-dim indices
            hard_token_idx = torch.where(hard_token_idx.flatten())[0]

            # topk
            # old_score = student_loss - target_loss
            # _, old_hard_token_idx = old_score.topk(k=int(old_score.numel() * loss_on_topk))
            # will be torch.allclose(old_score, score[score != -1e9])

    if hidden_factor > 0:
        for layer_id, (actual_states, target_states) in enumerate(
            zip(nc_outputs.hidden_states, target_hidden_states)
        ):
            actual_states = actual_states[nc_valid_idx, :]

            if actual_states.size(0) != target_states.size(0):
                # this shouldn't happen, but sometimes it does probably due to weird tokenization issues
                logger.warning("Skipping batch due to mismatch in shape")

            # Loss is the mean abs difference between target and predicted states,
            # normalised by mean magnitude of target states
            if loss_on_topk is not None:
                actual_states = actual_states[hard_token_idx, :]
                target_states = target_states[hard_token_idx, :]

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
    target_probs = F.softmax(target_logits / temp, dim=-1)
    preds = F.log_softmax(nc_outputs.logits[nc_valid_idx, ...] / temp, dim=-1)

    if loss_on_topk is not None:
        kl_loss = (
            kl_loss(preds[hard_token_idx], target_probs[hard_token_idx])
            .sum(dim=-1)
            .mean()
        )
    else:
        kl_loss = kl_loss(preds, target_probs).sum(dim=-1).mean()

    loss = loss + logit_factor * kl_loss
    return loss


def ema_dcd_loss(model, inputs, logit_factor=1.0, hidden_factor=1.0):
    """Deep Contextual Distillation loss."""
    kl_loss = torch.nn.KLDivLoss(reduction="none")

    # document + small task prompt + task output (e.g. summary, or question and answer)
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    valid_idx = labels != -100

    # small task prompt + task output (e.g. summary, or question and answer)
    nc_input_ids = inputs["nc_input_ids"]
    nc_labels = inputs["nc_labels"]
    nc_attention_mask = inputs["nc_attention_mask"]
    nc_valid_idx = nc_labels != -100

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

    # for the context-aware pass, use the EMA expert
    raw_model.set_default_expert("EMA")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        target_hidden_states = [
            hidden_state[valid_idx] for hidden_state in outputs.hidden_states
        ]
        target_logits = outputs.logits[valid_idx]

    # for the no context pass, use the KM expert
    raw_model.set_default_expert("KM")
    outputs = model(
        input_ids=nc_input_ids,
        attention_mask=nc_attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    # finally, revert to the model's default config
    raw_model.set_default_expert(raw_model.config.default_expert)

    loss = 0.0
    losses = []

    if hidden_factor > 0:
        for layer_id, (actual_states, target_states) in enumerate(
            zip(outputs.hidden_states, target_hidden_states)
        ):
            # actual_states = actual_states[nc_labels != -100, :]
            actual_states = actual_states[nc_valid_idx, :]

            if actual_states.size(0) != target_states.size(0):
                # this shouldn't happen, but sometimes it does probably due to weird tokenization issues
                logger.warning("Skipping batch due to mismatch in shape")

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
    preds = F.log_softmax(outputs.logits[nc_valid_idx, ...], dim=-1)
    kl_loss = kl_loss(preds, target_probs).sum(dim=-1).mean()

    loss = loss + logit_factor * kl_loss
    return loss


def lm_loss(model, inputs, prefix=""):
    """Next-token prediction loss."""

    assert prefix in ["", "nc_"]

    input_ids = inputs[f"{prefix}input_ids"]
    labels = inputs[f"{prefix}labels"]
    attention_mask = inputs[f"{prefix}attention_mask"]

    assert input_ids.size() == labels.size()
    # assert that labels is either -100 or the same as input_ids
    assert torch.all((labels == -100) | (labels == input_ids))

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        task_names=inputs.get("task_names"),
    )
    return outputs.loss


def do_evaluation(datamodule, model, loss_function, evaluator, **kwargs) -> bool:
    val_loss = []
    for batch in tqdm(datamodule.val_dataloader(), disable=not is_main_process()):
        with torch.no_grad():
            batch = transfer_batch_to_device(batch, model.device)
            val_loss.append(loss_function(model, batch).item())

    val_loss = distributed_mean(val_loss, model.device)
    if evaluator is not None:
        eval_score = evaluator.evaluate(model, "dev", **kwargs)
    else:
        eval_score = None
    return val_loss, eval_score


class SimpleLogger:
    def __init__(self, output_dir):
        self.output_file = os.path.join(output_dir, "metrics.json")
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def get_metric(self, metric_name):
        try:
            with open(self.output_file, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                lines = [l["value"] for l in lines if l["name"] == metric_name]
            return lines
        except:
            return None

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


class EarlyStopper:
    def __init__(self, patience, mode="min", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.best_score = float("inf")
        elif mode == "max":
            self.best_score = float("-inf")
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, score):
        if self.mode == "min":
            if score < self.best_score - self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == "max":
            if score > self.best_score + self.delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
