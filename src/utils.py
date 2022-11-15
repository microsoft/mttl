import glob
import json
import logging

import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)


def label_smoothed_nll_loss(lprobs, target, epsilon=0.1, ignore_index=-100):
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

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    # NOTE (Lucas): the original code does not divide by the batch size. Not great
    loss, nll_loss = loss / lprobs.size(0), nll_loss / lprobs.size(0)

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


def average_dicts(list_of_dicts):
    out = list_of_dicts[0]
    for item in list_of_dicts[1:]:
        assert len(item) == len(out)
        for k, v in item.items():
            out[k] += v

    return {k: v / len(list_of_dicts) for (k, v) in out.items()}


def get_checkpoint_path(path):
    # use glob to avoid explicitly writing out long paths

    match = glob.glob(f"{path}/**/*.ckpt", recursive=True)

    if len(match) > 1:
        logger.warning(
            f"{len(match)} checkpoints found. "
            + "taking the one with the lowest val loss"
        )
        idx = np.argmin([float(x.split("loss=")[-1].split(".ckpt")[0]) for x in match])
        path = match[idx]
    elif len(match) == 0:
        raise FileNotFoundError(f"{path} had no `.ckpt` files")
    else:
        path = match[0]

    return path
