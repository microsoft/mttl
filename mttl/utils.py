import glob
import hashlib
import logging
import os
import random
import string
from functools import lru_cache
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger("mttl")


@lru_cache
def warn_once(msg: str):
    logger.warning(msg)


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


def get_pl_loggers(args):
    loggers = []

    add_simple_logger(loggers, args)
    add_tb_logger(loggers, args)
    add_wandb_logger(loggers, args)
    add_mlf_logger(loggers)

    return loggers


def add_simple_logger(loggers, args):
    from mttl.models.utils import SimpleLogger

    loggers.append(SimpleLogger(args.output_dir))


def add_tb_logger(loggers, args):
    if args.tensorboard:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.output_dir)
        loggers.append(tb_logger)


def add_wandb_logger(loggers, args) -> pl.loggers.WandbLogger:
    if not os.environ.get("WANDB_API_KEY"):
        return

    import wandb

    if not args.exp_name:
        args.exp_name = os.environ.get("AMLT_JOB_NAME")
    if not args.wandb_project:
        args.wandb_project = os.environ.get("WANDB_PROJECT")

    wandb_logger = pl.loggers.WandbLogger(
        project=args.wandb_project,
        name=args.exp_name,
        settings=wandb.Settings(start_method="fork"),
    )
    wandb_logger.experiment.save("*.py")
    loggers.append(wandb_logger)


def add_mlf_logger(loggers):
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    class MLFlowLoggerCustom(pl.loggers.MLFlowLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @rank_zero_only
        def log_hyperparams(self, *args, **kwargs) -> None:
            try:
                super().log_hyperparams(*args, **kwargs)
            except:
                pass

    try:
        from azureml.core.run import Run

        run = Run.get_context()

        mlf_logger = MLFlowLoggerCustom(
            experiment_name=run.experiment.name,
        )
        mlf_logger._run_id = run.id
    except:
        logger.warning("Couldn't instantiate MLFlowLogger!")
        mlf_logger = None

    if mlf_logger is not None:
        loggers.append(mlf_logger)


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


def setup_logging(log_dir: str = None):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s --> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if log_dir:
        log_file_path = os.path.join(log_dir, "log.txt")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        handler_exists = any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == log_file_path
            for handler in logger.handlers
        )

        if not handler_exists:
            logger.addHandler(logging.FileHandler(log_file_path))
            logger.info(
                "New experiment, log will be at %s",
                log_file_path,
            )


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
