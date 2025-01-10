import math
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from openai import OpenAI
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

DEFAULT_TEMP = 0.5
DEFAULT_MAX_TOKENS = 768


class AccumulatorDict:
    """Accumulate values in a dictionary."""

    def __init__(self):
        self.data = {}

    def accumulate(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def mean(self, key):
        return sum(self.data[key]) / len(self.data[key])

    def get(self):
        # return mean values and clear the stats
        data = {k: sum(v) / len(v) for k, v in self.data.items()}
        self.data = {}
        return data

    def clear(self):
        self.data = {}


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=False, unbiased_variance=False):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(
        values, mask, unbiased=unbiased_variance
    )
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def trim_to_longest(sequences: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Trims each sequence in the batch to remove trailing pad tokens,
    returning a sub-tensor up to the longest real sequence length in the batch.

    Args:
        sequences (torch.Tensor): A 2D tensor of shape (batch_size, seq_len).
        pad_token_id (int): Token ID used for padding.

    Returns:
        torch.Tensor: A sub-tensor trimmed to the maximum sequence length (no trailing pads).
    """
    # Count non-pad tokens per sequence
    seq_lengths = (sequences != pad_token_id).sum(dim=1)
    max_len = seq_lengths.max().item()
    # Slice the tensor up to max_len
    return sequences[:, :max_len]


def get_shifted_logprobs(
    logits,
    labels,
    mask,
    temperature=DEFAULT_TEMP,
    reduction="none",
):
    logits /= temperature + 1e-7
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    logprobs = torch.gather(
        logits[:, :-1],
        2,
        labels[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    logprobs = logprobs * mask[:, 1:]
    if reduction == "mean":
        logprobs = logprobs.sum(dim=1) / mask[:, 1:].sum(dim=1)
    return logprobs


@torch.no_grad()
def get_logprobs(
    model,
    query_response,
    query_response_mask,
    response_mask,
    batch_size=4,
    reduction="mean",
    temperature=DEFAULT_TEMP,
):
    from accelerate.state import PartialState

    all_logprobs = []
    acc_state = PartialState()

    for batch in tqdm(
        range(0, len(query_response_mask), batch_size),
        desc=f"[{acc_state.local_process_index}] Gathering logprobs...",
        position=acc_state.local_process_index,
    ):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            mb_qm = query_response_mask[batch : batch + batch_size].to(acc_state.device)
            mb_q = query_response[batch : batch + batch_size].to(acc_state.device)
            mb_r = response_mask[batch : batch + batch_size].to(acc_state.device)
            output = model(
                input_ids=mb_q,
                attention_mask=mb_qm,
                return_dict=True,
            )
            logprobs = get_shifted_logprobs(
                output.logits, mb_q, mb_r, temperature=temperature, reduction=reduction
            )

        all_logprobs.append(logprobs.cpu())
        del output

    logprobs = torch.cat(all_logprobs, 0)
    torch.cuda.empty_cache()
    return logprobs


@torch.no_grad()
def get_entropies(
    model,
    query_response,
    query_response_mask,
    response_mask,
    batch_size=4,
    temperature=DEFAULT_TEMP,
    normalize=True,
):
    all_ents = []

    for batch in tqdm(
        range(0, len(query_response_mask), batch_size), desc="Gathering entropies..."
    ):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            mb_qm = query_response_mask[batch : batch + batch_size].to(model.device)
            mb_q = query_response[batch : batch + batch_size].to(model.device)
            mb_r = response_mask[batch : batch + batch_size].to(model.device)
            output = model(
                input_ids=mb_q,
                attention_mask=mb_qm,
                return_dict=True,
            )
            output = torch.softmax(output.logits / temperature, dim=-1)
            ents = -(output * torch.log(output + 1e-12)).sum(dim=-1)

            if normalize:
                ents = ents / math.log(output.shape[-1])

            ents = ents * mb_r
            ents = ents.cpu()

            for ent in ents:
                all_ents.append(ent.tolist())  # Convert to list and extend
        del output

    return all_ents


def compute_kl_divergence(ref_logprobs, logprobs, mask=None, kl_min=0, kl_max=10):
    """Approx to KL divergence between two distributions."""
    assert ref_logprobs.ndim == 2 and logprobs.ndim == 2
    assert ref_logprobs.shape == logprobs.shape

    log_ratio = ref_logprobs - logprobs
    ref_kl = torch.exp(log_ratio) - log_ratio - 1

    if mask is not None:
        ref_kl = ref_kl * mask

    ref_kl = torch.clamp(
        ref_kl,
        min=kl_min,
        max=kl_max,
    )
    ref_kl_loss = ref_kl.sum(dim=1).mean()
    return ref_kl_loss


def get_wsd_scheduler(
    optimizer,
    total_steps,
    min_lr_ratio=0.0,
    last_epoch=-1,
):
    from torch.optim.lr_scheduler import LambdaLR

    # Calculate step counts from proportions
    num_warmup_steps = int(total_steps * 0.01)
    num_stable_steps = int(total_steps * 0.89)
    num_decay_steps = total_steps - num_warmup_steps - num_stable_steps

    def get_ratio(
        current_step, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio
    ):
        if current_step < num_warmup_steps:
            return (
                current_step / float(num_warmup_steps) if num_warmup_steps > 0 else 1.0
            )
        elif current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        elif current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
            steps_in_decay = num_warmup_steps + num_stable_steps
            progress = (
                (current_step - steps_in_decay) / float(num_decay_steps)
                if num_decay_steps > 0
                else 1.0
            )
            return 1.0 - (1.0 - min_lr_ratio) * progress
        return min_lr_ratio

    def lr_lambda(step):
        return get_ratio(
            step,
            num_warmup_steps,
            num_stable_steps,
            num_decay_steps,
            min_lr_ratio,
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_ending_tokens(tokenizer):
    """Get the ending tokens for the chat template of the tokenizer."""
    dummy_template = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "dummy content"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return dummy_template[
        dummy_template.rindex("dummy content") + len("dummy content") :
    ]


def print_metrics(data, min_value=None, max_value=None):
    import numpy as np

    spark_chars = "▁▂▃▄▅▆▇█"

    if min_value is None and max_value is None:
        min_value = min(data)
        max_value = max(data)

        # Handle the case where all data points are the same
        if max_value - min_value == 0:
            return spark_chars[3] * len(data)

        std = np.std(data)
        if std == 0:
            return "<std = 0>, skipping"

        min_value = min(data) - std
        max_value = max(data) + std
    else:
        assert max_value != min_value

    # Scale data points to indices of spark_chars
    scaled_data = [
        int((value - min_value) / (max_value - min_value) * (len(spark_chars) - 1))
        for value in data
    ]

    # Map scaled data to corresponding sparkline characters
    return "".join(spark_chars[idx] for idx in scaled_data)


class CosineWarmupScheduler(_LRScheduler):
    def __init__(
        self, optimizer, max_lr, min_lr, warmup_steps, max_steps, last_epoch=-1
    ):
        """
        Initializes the CosineWarmupScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_lr (float): The peak learning rate.
            min_lr (float): The minimum learning rate after decay.
            warmup_steps (int): Number of steps for linear warmup.
            max_steps (int): Total number of steps for the scheduler.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Computes the learning rate for each parameter group.

        Returns:
            List[float]: A list of learning rates for each parameter group.
        """
        step = self.last_epoch + 1  # Current step (starts from 1)
        new_lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Linear warmup
                lr = self.max_lr * (step + 1) / self.warmup_steps
            elif step > self.max_steps:
                # After max_steps, keep min_lr
                lr = self.min_lr
            else:
                # Cosine decay
                decay_ratio = (step - self.warmup_steps) / (
                    self.max_steps - self.warmup_steps
                )
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
            new_lrs.append(lr)
        return new_lrs


def setup_output_directory(output_dir: str):
    import os
    import shutil

    from ddp_utils import ddp_state

    if ddp_state.is_master:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir)
        form_code_zip(output_dir)

    if ddp_state.ddp:
        torch.distributed.barrier()

    return output_dir


def save_args(args, output_dir: str):
    import json
    import os

    from ddp_utils import ddp_state

    if ddp_state.is_master:
        with open(os.path.join(output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)


def repeat(ary: List[Any], k):
    import copy

    return [copy.deepcopy(a) for a in ary for _ in range(k)]


def flatten(ary: List[List[Any]]):
    return [r for sublist in ary for r in sublist]


def form_code_zip(output_dir):
    import subprocess
    import zipfile

    try:
        tracked_files = (
            subprocess.check_output(["git", "ls-files"]).decode().splitlines()
        )

        with zipfile.ZipFile(
            output_dir + "/codebase.zip", "w", zipfile.ZIP_DEFLATED
        ) as zf:
            for f in tracked_files:
                zf.write(f)
    except Exception as e:
        print("Could not form codebase.zip!", e)
