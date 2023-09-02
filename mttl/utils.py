from collections import defaultdict
import glob
import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict
import hashlib
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.autograd.function import Function


logger = logging.getLogger("mttl")


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
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

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
                self.total[key] = self.total[key] * self.weight + value * (1 - self.weight)
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


def get_mlf_logger():
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
        mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
        mlf_logger = MLFlowLoggerCustom(
            experiment_name=run.experiment.name, tracking_uri=mlflow_url
        )
        mlf_logger._run_id = run.id
    except:
        mlf_logger = None
    return mlf_logger


def get_checkpoint_path(path, step=None, use_last=False):
    if path.endswith(".ckpt") or path.endswith(".pt"):
        return path

    # use glob to avoid explicitly writing out long paths
    match = glob.glob(f"{path}/*.ckpt", recursive=True)

    if use_last:
        # search for last.ckpt
        match = [m for m in match if "last.ckpt" in m]
        if len(match) != 1:
            raise ValueError(
                "last.ckpt not found or found multiple (?) in the list of checkpoints!"
            )
        return match[0]

    if len(match) > 1:
        logger.warning(
            f"{len(match)} checkpoints found. "
            + "taking the one with the lowest val loss"
        )
        losses = []
        for x in match:
            if "loss" in x:
                loss = float(x.split("loss=")[-1].split(".ckpt")[0])
            elif "zero_shot_perf" in x:
                loss = -float(x.split("zero_shot_perf=")[-1].split(".ckpt")[0])
            else:
                continue
            losses.append(loss)
        idx = np.argmin(losses) if losses else 0
        path = match[idx]
    elif len(match) == 0:
        match = glob.glob(f"{path}/*step*.pt", recursive=True)
        if len(match) > 1:
            logger.warning(
                f"{len(match)} checkpoints found. "
                + "taking the one with the lowest val loss"
            )
            found = False
            for m in match:
                # take the one with the specified step
                if str(step) in m:
                    path = m
                    found = True
                    break
            if not found and step is None:
                # global_stepX.pt, take the one with the highest step
                idx = np.argmax(
                    [float(x.split("step")[-1].split(".pt")[0]) for x in match]
                )
                path = match[idx]
        elif len(match) == 0:
            raise FileNotFoundError(f"{path} had no `.ckpt` nor `.pt` files")
        else:
            path = match[0]
    else:
        path = match[0]

    print("Found checkpoint", path)
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
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger.addHandler(logging.FileHandler(os.path.join(log_dir, "log.txt")))
        logger.info(
            "New experiment, log will be at %s",
            os.path.join(log_dir, "log.txt"),
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
