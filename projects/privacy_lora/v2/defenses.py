"""AdaptGuard (D5) distillation and helpers for shard training (D4)."""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import torch
from tqdm.auto import tqdm

from projects.privacy_lora.v2.attacks import collate, encode_example
from projects.privacy_lora.v2.data import Example


def distill_adaptguard(teacher, student, tok, public_examples: Sequence[Example], device: str, max_length: int,
                       n_steps: int = 600, micro_batch: int = 4, lr: float = 1e-4, top_k: int = 50,
                       logit_noise: float = 0.5, temperature: float = 1.0, pub_only: bool = False, seed: int = 0) -> Dict:
    """Fit ``student`` (fresh adapter) on D_pub by distillation from the privatised ``teacher``.

    pub_only=True is the ablation "train a fresh adapter on D_pub with the LM loss, no teacher".
    top_k<=0 disables truncation; logit_noise=0 disables noise.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    enc = [encode_example(tok, ex, max_length) for ex in public_examples]
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    student.train()
    if teacher is not None:
        teacher.eval()
    step = 0
    losses = []
    n_queries = 0
    pbar = tqdm(total=n_steps, desc="adaptguard", leave=False)
    while step < n_steps:
        order = list(range(len(enc)))
        rng.shuffle(order)
        for s in range(0, len(order), micro_batch):
            idx = order[s : s + micro_batch]
            input_ids, attn, labels = collate([enc[i] for i in idx], tok.pad_token_id, device)
            if pub_only:
                loss = student(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            else:
                with torch.no_grad():
                    t_logits = teacher(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float() / temperature
                    if top_k and top_k < t_logits.size(-1):
                        vals, ids = torch.topk(t_logits, k=top_k, dim=-1)
                        masked = torch.full_like(t_logits, float("-inf"))
                        masked.scatter_(-1, ids, vals)
                        t_logits = masked
                    if logit_noise > 0:
                        noise = torch.randn_like(t_logits) * logit_noise
                        t_logits = t_logits + noise
                    t_probs = torch.softmax(t_logits, dim=-1)
                    n_queries += len(idx)
                s_logp = torch.log_softmax(student(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float(), dim=-1)
                mask = (labels[:, 1:] != -100).float()
                kl = (t_probs * (torch.log(t_probs.clamp_min(1e-12)) - s_logp)).sum(-1)
                loss = (kl * mask).sum() / mask.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss))
            step += 1
            pbar.update(1)
            if step >= n_steps:
                break
    pbar.close()
    student.eval()
    return {"steps": step, "final_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1), "n_public": len(public_examples),
            "teacher_queries": n_queries, "top_k": top_k, "logit_noise": logit_noise, "pub_only": pub_only}
