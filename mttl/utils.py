import glob
import json
import logging
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Optional
import hashlib
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch.autograd.function import Function

logger = logging.getLogger("mttl")


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


def template_to_string(template):
    return template.jinja + (
        (" answer_choices: " + template.answer_choices)
        if template.answer_choices
        else ""
    )


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


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def get_tasks_list(filename, split_name):
    with open(filename, "r") as fin:
        split_dict = json.load(fin)
    return split_dict[split_name]


def get_ni_tasks_from_file(filename):
    with open(filename, "r") as f:
        tasks = f.readlines()
        tasks = [task.strip() for task in tasks]
    task2id = {task: idx for idx, task in enumerate(tasks)}
    return tasks, task2id


def get_example_to_ids(filename):
    import pickle

    with open(filename, "rb") as f:
        package = pickle.load(f)
    return package


class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.total = {}

    def update(self, stats):
        for key, value in stats.items():
            if key not in self.total:
                self.total[key] = value
            else:
                self.total[key] = self.total[key] * self.weight + value * (
                    1 - self.weight
                )
        return self.total


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


class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_model_score = None

    def _update_best_and_save(
        self,
        current: Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, Tensor],
    ) -> None:
        """First remove checkpoint, THEN save it."""
        import os

        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"), device=current.device
            )

        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates, trainer, del_filepath
        )

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )

        if del_filepath is not None and filepath != del_filepath:
            print(f"Removing checkpoint... {del_filepath}")
            trainer.strategy.remove_checkpoint(del_filepath)

        print(f"Saving checkpoint... {filepath}")
        self._save_checkpoint(trainer, filepath)
        os.system("df")
        os.system(f"ls -al {filepath}")

        self.last_model_score = current


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


class MemEfficientLoRA(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, A, B, skills):
        ctx.save_for_backward(input, A, B, skills)

        bs, L, I = input.size()
        bs, n_skills = skills.size()
        n_skills, I, R = A.size()
        n_skills, R, O = B.size()

        output = torch.einsum("BLI,BS,SIR,SRO->BLO", (input, skills, A, B))

        return output

    @staticmethod
    def backward(ctx, O_grad):
        bs, L, O = O_grad.size()

        input, A, B, skills = ctx.saved_tensors

        bs, S, I = input.size()
        bs, n_skills = skills.size()
        n_skills, I, R = A.size()
        n_skills, R, O = B.size()

        # option A)
        W = torch.einsum("BS,SIR,SRO->BIO", (skills, A, B))
        I_grad = torch.einsum("BLO,BIO->BLI", (O_grad, W))

        # option B) [OOM]
        # I_grad = torch.einsum('BLO,BS,SIR,SRO->BLI', (O_grad, skills, A, B))

        W_grad = torch.einsum("BLO,BLI->BIO", (O_grad, input))

        tmp = torch.einsum("BIO,BS->SIO", (W_grad, skills))
        A_grad = torch.einsum("SIO,SRO->SIR", (tmp, B))
        B_grad = torch.einsum("SIO,SIR->SRO", (tmp, A))
        S_grad = torch.einsum("BIO,SIR,SRO->BS", (W_grad, A, B))

        return I_grad, A_grad, B_grad, S_grad


if __name__ == "__main__":
    B, L, S, I, O, R = 3, 5, 8, 3, 12, 4
    fn = MemEfficientLoRA.apply

    for i in range(10):
        input = torch.randn(B, L, I, dtype=torch.double).cuda()
        Am = torch.randn(S, I, R, dtype=torch.double).cuda()
        Bm = torch.randn(S, R, O, dtype=torch.double).cuda()
        skill = torch.randn(B, S, dtype=torch.double).cuda()
        idx1 = torch.multinomial(torch.ones(S, I * O).cuda(), num_samples=10)
        idx2 = torch.arange(S).repeat_interleave(10).cuda()
        idx = torch.stack([idx1.flatten(), idx2.flatten()])
        val = torch.randn(size=idx.shape[1:]).cuda().double()
        W = torch.sparse_coo_tensor(idx, val, (I * O, S)).coalesce()

        # coll = [input, Am, Bm, skill]
        coll = [input, W, skill]
        for x in coll:
            x.requires_grad = True

        res = torch.autograd.gradcheck(fn, coll, check_sparse_nnz=True)
        import pdb

        pdb.set_trace()
        print(res)


# define a retry decorator
def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (),
):
    """Retry a function with exponential backoff."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors as e:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator


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
