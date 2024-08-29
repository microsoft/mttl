import functools
import glob
import hashlib
import os
import random
import string
from datetime import time
from functools import wraps
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from mttl.logging import logger, warn_once


def retry(max_retries=10, wait_seconds=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:  # requests.exceptions.HTTPError as e:
                    print(e, type(e), "retrying...")
                    if attempt < max_retries:
                        print(f"Waiting {wait_seconds} seconds before retrying...")
                        time.sleep(wait_seconds)
            raise RuntimeError(
                f"Function {wrapper.__name__} failed after {max_retries} attempts."
            )

        return wrapper

    return decorator


def remote_login(token: Optional[str] = None):
    """Caches the provided token and login to remote service: Azure Blob Storage or Hugging Face Hub.

    Sets the environment variable "BLOB_SAS_TOKEN" for later use
    if token is provided and it does not start with "hf_"

    Otherwise, Hugging Face Hub login is performed. If no token is provided,
    tries to login to Hugging Face Hub using HF_TOKEN environment variable.

    Args:
        token (str): The token to use for login.

    Returns:
        None
    """

    if token is not None and not token.startswith("hf_"):
        os.environ["BLOB_SAS_TOKEN"] = token
    else:
        token = token or os.environ.get("HF_TOKEN", None)
        if token is not None:
            from huggingface_hub import login as hf_hub_login

            hf_hub_login(token=token)


def hash_example(example):
    return hashlib.md5(example.encode("utf-8")).hexdigest()


def label_smoothed_nll_loss(
    lprobs, target, epsilon=0.1, ignore_index=-100, reduction="mean"
):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    pad_mask = target.eq(ignore_index)
    # because otherwise we gather -100 :-(
    target.masked_fill_(pad_mask, 0)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    nll_loss.masked_fill_(pad_mask, 0.0)
    smooth_loss.masked_fill_(pad_mask, 0.0)

    if reduction == "mean":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

        # NOTE (Lucas): the original code does not divide by the batch size. Not great
        loss, nll_loss = loss / lprobs.size(0), nll_loss / lprobs.size(0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def agg_dicts(list_of_dicts, agg="mean", tag=False):
    """Aggregate a list of dicts by taking the aggregate of each key.

    Could be "min", "max", or "mean".
    """
    out = {}
    for curr_dict in list_of_dicts:
        for k, v in curr_dict.items():
            if tag:
                k = f"{agg}_{k}"
            if k not in out:
                # clone the variable so that we don't modify the original
                out[k] = v.clone() if isinstance(v, torch.Tensor) else v
            else:
                if agg == "min":
                    # take minimum between tensors
                    out[k] = (
                        torch.minimum(out[k], v)
                        if isinstance(v, torch.Tensor)
                        else min(out[k], v)
                    )
                elif agg == "max":
                    out[k] = (
                        torch.maximum(out[k], v)
                        if isinstance(v, torch.Tensor)
                        else max(out[k], v)
                    )
                else:
                    out[k] += v
    if agg == "mean":
        for k, v in out.items():
            out[k] = v / len(list_of_dicts)
    return out


def deprecated(message=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warning_msg = (
                message
                if message
                else f"{func.__name__}() is deprecated and will be removed in a future version."
            )
            warn_once(warning_msg)
            return func(*args, **kwargs)

        return wrapped

    return decorator


def get_checkpoint_path(path, step=None, use_last=False):
    if path.endswith(".ckpt") or path.endswith(".pt"):
        return path

    # use glob to avoid explicitly writing out long paths
    matches = glob.glob(f"{path}/**/*.ckpt", recursive=True)

    if use_last:
        # search for last.ckpt
        match = [m for m in matches if "last.ckpt" in m]
        if len(match) != 1:
            raise ValueError(
                "last.ckpt not found or found multiple (?) in the list of checkpoints!"
            )
        path = match[0]
    else:
        # match the filename
        match = [m for m in matches if "best" in m.split("/")[-1]]
        if len(match) == 0:
            logger.warning("No best checkpoints found! Defaulting to 'last'.")

            match = [m for m in matches if "last" in m]
            path = match[0]
        elif len(match) > 1:
            logger.warning(
                "Multiple best checkpoints found! Taking the most recent one!"
            )
            logger.warning(match)
            path = max(match, key=os.path.getctime)
        else:
            path = match[0]

    logger.info(f"Found checkpoint at {path}.")
    return path


# decorator like rank_zero_only but with a barrier at the end
def rank_zero_only_and_wait(before=True, after=True):
    def decorator(fn):
        def wrapped_fn(*args, **kwargs):
            output = None
            if (
                before
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.barrier()
            if rank_zero_only.rank == 0:
                output = fn(*args, **kwargs)
            if (
                after
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.barrier()

            return output

        return wrapped_fn

    return decorator


def generate_random_string(str_len=10):
    return "".join(random.choices(string.ascii_uppercase, k=str_len))
