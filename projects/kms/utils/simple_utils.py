import json
import os
from contextlib import contextmanager

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
from mttl.models.utils import compute_loglike_loss, transfer_batch_to_device


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


def mc_loss(
    model,
    inputs,
):
    """
    Multiple choice training loss, we normalize the log-likelihood of each answer,
    and compute cross-entropy on the correct label.
    """
    import numpy as np

    input_ids = inputs["input_ids"]
    labels = inputs[f"labels"]
    attention_mask = inputs[f"attention_mask"]
    num_options = inputs["num_options"]
    labels_index = inputs["labels_index"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_names=inputs.get("task_names"),
    )
    loss_per_option = compute_loglike_loss(
        outputs.logits,
        labels,
        reduction="none",
        normalize_length=True,
    )
    del outputs, input_ids, attention_mask
    torch.cuda.empty_cache()

    loss_per_example = [
        loss_per_option[
            int(np.sum(num_options[:i])) : int(np.sum(num_options[: i + 1]))
        ]
        for i in range(len(labels_index))
    ]
    loss_per_example = [
        -torch.log_softmax(-option_losses, dim=0)[labels_index[i]]
        for i, option_losses in enumerate(loss_per_example)
    ]
    return torch.stack(loss_per_example).mean()


def mc_loss_iterative(
    model,
    inputs,
):
    """
    Multiple choice training loss, we normalize the log-likelihood of each answer,
    and compute cross-entropy on the correct label.

    Optimized to avoid redundant computation on shared prefixes in multiple choice questions.
    """
    import numpy as np
    import torch

    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    attention_mask = inputs["attention_mask"]
    num_options = inputs["num_options"]
    labels_index = inputs["labels_index"]

    # Find the shared prefixes for each question
    batch_size = len(num_options)
    start_indices = np.cumsum([0] + num_options[:-1])

    # Process each question group separately
    loss_per_example = []

    for i in range(batch_size):
        options_start_idx = start_indices[i]
        options_end_idx = options_start_idx + num_options[i]

        # Get all options for this question
        x_input_ids = input_ids[options_start_idx:options_end_idx]
        x_attn_mask = attention_mask[options_start_idx:options_end_idx]
        x_labels = labels[options_start_idx:options_end_idx]

        # Find the shared prefix (assuming the first tokens are identical across options)
        # This is a simplification - a real implementation would need to find the exact
        # shared prefix length for each question
        is_same = (x_input_ids[0] == x_input_ids[1:]).all(0)
        prefix_length = torch.where(~is_same)[0].min()

        prefix_outputs = model(
            input_ids=x_input_ids[
                [0], :prefix_length
            ],  # Just use the first one since they're identical
            attention_mask=x_attn_mask[[0], :prefix_length],
            task_names=inputs.get("task_names")[
                options_start_idx : options_start_idx + 1
            ],
            use_cache=True,
            return_dict=True,
        )

        # Extract past key values
        past_key_values = prefix_outputs.past_key_values

        # Second pass: process the unique completion for each option
        option_losses = []
        for j in range(num_options[i]):
            suffix_ids = x_input_ids[[j], prefix_length:]
            suffix_labels = x_labels[[j], prefix_length:]

            # Only process the unique part using the cached prefix
            suffix_outputs = model(
                input_ids=suffix_ids,
                attention_mask=x_attn_mask[[j]],
                past_key_values=past_key_values,
                task_names=inputs.get("task_names")[
                    options_start_idx : options_start_idx + 1
                ],
            )

            # Compute loss for this option
            suffix_loss = compute_loglike_loss(
                suffix_outputs.logits,
                suffix_labels.unsqueeze(0),
                reduction="none",
                normalize_length=True,
            )
            option_losses.append(suffix_loss)

        option_losses = torch.cat(option_losses)
        # build a distribution over all options
        example_loss = -torch.log_softmax(-option_losses, dim=0)[labels_index[i]]

        # instead of averaging later over the batch_size, we divide the loss now
        example_loss = example_loss / batch_size

        if (i + 1) == batch_size:
            # for the last example, we don't backward, leave it to the backward
            # call in the main file
            loss_per_example += [example_loss]
        else:
            # for the other examples, we need to backward
            example_loss.backward()
            loss_per_example += [example_loss.detach()]

    return torch.stack(loss_per_example).sum()


def dcd_loss(
    model,
    inputs,
    logit_factor=1.0,
    hidden_factor=1.0,
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
        del outputs.logits
        torch.cuda.empty_cache()

    nc_outputs = model(
        input_ids=nc_input_ids,
        attention_mask=nc_attention_mask,
        output_hidden_states=True,
        return_dict=True,
        task_names=inputs.get("task_names"),
    )

    loss = 0.0
    losses = []

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


def do_evaluation(
    datamodule,
    model,
    loss_function,
    evaluator=None,
    evaluator_split="dev",
    split="dev",
    **kwargs,
) -> bool:

    state = model.training
    model.eval()
    if evaluator is not None:
        eval_score = evaluator.evaluate(model, split=evaluator_split, **kwargs)
    else:
        eval_score = None

    if split == "dev":
        eval_dataloader = datamodule.val_dataloader()
    else:
        eval_dataloader = datamodule.test_dataloader()

    eval_loss = []
    for batch in tqdm(eval_dataloader, disable=not is_main_process()):
        with torch.no_grad():
            batch = transfer_batch_to_device(batch, model.device)
            eval_loss.append(loss_function(model, batch).item())
            del batch
    torch.cuda.empty_cache()

    eval_loss = distributed_mean(eval_loss, model.device)

    torch.cuda.empty_cache()
    model.train(state)
    return eval_loss, eval_score


class SimpleLogger:
    def __init__(self, output_dir, file="metrics.json"):
        self.output_file = os.path.join(output_dir, file)

        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)

            try:
                if os.path.exists(self.output_file):
                    with open(self.output_file, "w") as f:
                        pass
            except:
                pass

    def get_metric(self, metric_name):
        try:
            with open(self.output_file, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                lines = [l["value"] for l in lines if l["name"] == metric_name]
            return lines
        except:
            return None

    def log_metrics(self, metrics, step=None):
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


@contextmanager
def cpu_offload(model, names, enable=False):
    """Swap the specified set of KMs from CPU to GPU."""
    if enable:
        names = set(names)
        if hasattr(model, "experts_containers"):
            for container in model.experts_containers:
                for name in names:
                    device = model.device
                    requires_grad = container.lora_a[name].requires_grad
                    container.lora_a[name] = container.lora_a[name].to(device)
                    container.lora_b[name] = container.lora_b[name].to(device)
                    container.lora_a[name].requires_grad = requires_grad
                    container.lora_b[name].requires_grad = requires_grad
        torch.cuda.empty_cache()
    yield
    if enable:
        if hasattr(model, "experts_containers"):
            for container in model.experts_containers:
                for name in names:
                    requires_grad = container.lora_a[name].requires_grad
                    container.lora_a[name] = container.lora_a[name].to("cpu")
                    container.lora_b[name] = container.lora_b[name].to("cpu")
                    container.lora_a[name].requires_grad = requires_grad
                    container.lora_b[name].requires_grad = requires_grad
        torch.cuda.empty_cache()
