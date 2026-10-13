"""Training loop with per-epoch callbacks, optional DP-SGD (per-sample clipping + Poisson sampling)."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from projects.privacy_lora.v2.attacks import collate, encode_example
from projects.privacy_lora.v2.data import Example
from projects.privacy_lora.v2.dp import epsilon_for, noise_multiplier_for


@dataclass
class TrainSpec:
    epochs: int = 3
    lr: float = 1e-4
    batch: int = 8            # logical batch size
    micro_batch: int = 1      # physical batch size
    optimizer: str = "adamw"  # adamw | sgd | adamw8bit
    weight_decay: float = 0.0
    max_length: int = 384
    seed: int = 0
    clip: float = 1.0
    eval_epochs: Sequence[int] = field(default_factory=lambda: [1, 2, 3])
    max_steps: Optional[int] = None
    # DP-SGD
    dp_epsilon: Optional[float] = None
    dp_delta: Optional[float] = None
    dp_clip: float = 1.0
    dp_batch: int = 256


def make_optimizer(params, spec: TrainSpec):
    if spec.optimizer == "sgd":
        return torch.optim.SGD(params, lr=spec.lr, momentum=0.9)
    if spec.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=spec.lr, weight_decay=spec.weight_decay)
    return torch.optim.AdamW(params, lr=spec.lr, weight_decay=spec.weight_decay)


def train_model(model, tok, examples: Sequence[Example], spec: TrainSpec, device: str,
                on_epoch_end: Optional[Callable[[int, Dict], None]] = None,
                val_examples: Optional[Sequence[Example]] = None) -> Dict:
    torch.manual_seed(spec.seed)
    rng = random.Random(spec.seed)
    enc = [encode_example(tok, ex, spec.max_length) for ex in examples]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, spec)
    n = len(enc)
    history: List[Dict] = []
    t0 = time.time()
    info: Dict = {"n_examples": n, "n_trainable": sum(p.numel() for p in params)}

    def val_nll() -> Optional[float]:
        if not val_examples:
            return None
        from projects.privacy_lora.v2.attacks import score_examples

        model.eval()
        sc = score_examples(model, tok, val_examples, spec.max_length, device, desc="val")
        model.train()
        return float(sc["loss"].mean())

    model.train()
    if spec.dp_epsilon is not None:
        # ---------------- DP-SGD ----------------
        q = spec.dp_batch / n
        steps_per_epoch = int(round(1.0 / q))
        total_steps = steps_per_epoch * spec.epochs
        delta = spec.dp_delta if spec.dp_delta is not None else 1.0 / (n**1.1)
        sigma = noise_multiplier_for(spec.dp_epsilon, q, total_steps, delta)
        info.update({"dp": {"epsilon_target": spec.dp_epsilon, "delta": delta, "sample_rate": q, "steps": total_steps,
                            "noise_multiplier": sigma, "clip": spec.dp_clip,
                            "epsilon_spent": epsilon_for(sigma, q, total_steps, delta)}})
        step = 0
        for epoch in range(1, spec.epochs + 1):
            losses = []
            for _ in tqdm(range(steps_per_epoch), desc=f"dp epoch {epoch}", leave=False):
                idx = [i for i in range(n) if rng.random() < q]  # Poisson sampling
                acc = [torch.zeros_like(p, dtype=torch.float32) for p in params]
                for i in idx:
                    input_ids, attn, labels = collate([enc[i]], tok.pad_token_id, device)
                    loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
                    grads = torch.autograd.grad(loss, params, allow_unused=True)
                    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
                    norm = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
                    factor = min(1.0, spec.dp_clip / (float(norm) + 1e-6))
                    for a, g in zip(acc, grads):
                        a.add_(g.float(), alpha=factor)
                    losses.append(float(loss))
                for p, a in zip(params, acc):
                    a.add_(torch.randn_like(a) * (sigma * spec.dp_clip))
                    p.grad = (a / spec.dp_batch).to(p.dtype)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                            "val_nll": val_nll(), "step": step})
            if on_epoch_end and epoch in set(spec.eval_epochs):
                on_epoch_end(epoch, history[-1])
    else:
        # ---------------- standard ----------------
        accum = max(1, spec.batch // spec.micro_batch)
        step = 0
        done = False
        for epoch in range(1, spec.epochs + 1):
            order = list(range(n))
            rng.shuffle(order)
            losses = []
            opt.zero_grad(set_to_none=True)
            n_micro = math.ceil(n / spec.micro_batch)
            for mb in tqdm(range(n_micro), desc=f"epoch {epoch}", leave=False):
                idx = order[mb * spec.micro_batch : (mb + 1) * spec.micro_batch]
                input_ids, attn, labels = collate([enc[i] for i in idx], tok.pad_token_id, device)
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
                (loss / accum).backward()
                losses.append(float(loss))
                if (mb + 1) % accum == 0 or mb + 1 == n_micro:
                    torch.nn.utils.clip_grad_norm_(params, spec.clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if spec.max_steps and step >= spec.max_steps:
                        done = True
                        break
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_nll": val_nll(), "step": step})
            if on_epoch_end and (epoch in set(spec.eval_epochs) or done):
                on_epoch_end(epoch, history[-1])
            if done:
                break
    info["history"] = history
    info["train_seconds"] = time.time() - t0
    model.eval()
    return info
