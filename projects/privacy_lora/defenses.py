"""Defenses D1–D5 for released LoRA adapters."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from projects.privacy_lora.train_adapter import (
    NoteDataset,
    TrainConfig,
    attach_lora,
    collect_lora_modules,
    load_base_model,
    load_lora_state,
    lora_state_dict,
    truncate_lora_rank,
)


def defense_d2_truncate(model, keep_rank: int = 2):
    truncate_lora_rank(model, keep_rank)
    return model


@torch.no_grad()
def defense_d4_merge_state_dicts(
    state_dicts: Sequence[Dict[str, torch.Tensor]],
    method: str = "ties",
    densify: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Merge multiple LoRA state dicts (uniform / TIES-style)."""
    keys = state_dicts[0].keys()
    merged = {}
    for key in keys:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        if method == "uniform":
            merged[key] = stacked.mean(dim=0)
            continue
        # TIES-like: trim small magnitudes per adapter, elect sign, average
        flat = stacked.reshape(stacked.size(0), -1)
        k = max(1, int(densify * flat.size(1)))
        # keep top-k |w| per expert
        mag = flat.abs()
        topk = torch.topk(mag, k=k, dim=1)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, topk.indices, 1.0)
        trimmed = flat * mask
        # sign election
        sign = torch.sign(trimmed.sum(dim=0, keepdim=True))
        sign[sign == 0] = 1
        agreed = trimmed * (torch.sign(trimmed) == sign).float()
        denom = (agreed != 0).float().sum(dim=0).clamp_min(1.0)
        avg = agreed.sum(dim=0) / denom
        merged[key] = avg.reshape_as(stacked[0]).to(state_dicts[0][key].dtype)
    return merged


def defense_d5_adaptguard(
    teacher_model,
    tokenizer,
    public_texts: Sequence[str],
    cfg: TrainConfig,
    device: str,
    n_steps: int = 200,
    top_k: int = 50,
    logit_noise: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Re-fit a fresh adapter on public data by privatized distillation from teacher.

    Teacher logits are truncated to top-k and corrupted with Gaussian noise
    before the student KL, so the released student depends on private data
    only through this privatized output channel.
    """
    student, _ = load_base_model(cfg.model_name, device)
    attach_lora(student, cfg)
    student.train()
    teacher_model.eval()

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
    ds = NoteDataset(public_texts, tokenizer, cfg.max_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    step = 0
    pbar = tqdm(total=n_steps, desc="adaptguard")
    while step < n_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                t_out = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                t_logits = t_out.logits
                # top-k truncation + noise
                if top_k and top_k < t_logits.size(-1):
                    vals, idx = torch.topk(t_logits, k=top_k, dim=-1)
                    masked = torch.full_like(t_logits, float("-inf"))
                    masked.scatter_(-1, idx, vals)
                    t_logits = masked
                t_logits = t_logits + logit_noise * torch.randn_like(t_logits)
                t_probs = torch.softmax(t_logits, dim=-1)

            s_out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            s_logprobs = torch.log_softmax(s_out.logits, dim=-1)
            # token-level KL
            kl = torch.nn.functional.kl_div(s_logprobs, t_probs, reduction="batchmean")
            opt.zero_grad()
            kl.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            pbar.update(1)
            if step >= n_steps:
                break
    pbar.close()
    state = lora_state_dict(student)
    del student
    torch.cuda.empty_cache()
    return state
