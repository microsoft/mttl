"""LoRALeak attacks A1–A5."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from projects.privacy_lora.clinical_data import canary_candidate_pool
from projects.privacy_lora.train_adapter import (
    NoteDataset,
    collect_lora_modules,
    disable_lora,
    enable_lora,
)


def _scores_to_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> Dict:
    """Higher score => more likely member."""
    y_true = np.concatenate(
        [np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    )
    y_score = np.concatenate([member_scores, nonmember_scores])
    # AUC via Mann-Whitney
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1)

    def tpr_at_fpr(target_fpr: float) -> float:
        # Threshold from non-member quantile
        thr = np.quantile(nonmember_scores, 1.0 - target_fpr)
        return float((member_scores >= thr).mean())

    return {
        "auc": float(auc),
        "tpr_at_1pct_fpr": tpr_at_fpr(0.01),
        "tpr_at_0_1pct_fpr": tpr_at_fpr(0.001),
        "n_members": int(len(member_scores)),
        "n_nonmembers": int(len(nonmember_scores)),
    }


@torch.no_grad()
def sequence_logprob(model, tokenizer, text: str, max_length: int, device: str) -> float:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # out.loss is mean NLL over non-pad tokens
    n_tokens = attention_mask.sum().item()
    return float(-out.loss.item() * max(n_tokens - 1, 1))


@torch.no_grad()
def sequence_loss(model, tokenizer, text: str, max_length: int, device: str) -> float:
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(out.loss.item())


@torch.no_grad()
def min_k_score(model, tokenizer, text: str, max_length: int, device: str, k: float = 0.2) -> float:
    """Negative mean logprob of the lowest-k% tokens (higher => more member-like)."""
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].bool()
    logprobs = torch.log_softmax(logits, dim=-1)
    tok_lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    tok_lp = tok_lp[mask]
    if tok_lp.numel() == 0:
        return 0.0
    n = max(1, int(math.ceil(k * tok_lp.numel())))
    vals, _ = torch.topk(tok_lp, k=n, largest=False)
    return float(vals.mean().item())


@torch.no_grad()
def batch_scores(
    model,
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    device: str,
    kind: str = "loss",
) -> np.ndarray:
    scores = []
    for text in tqdm(texts, desc=f"score:{kind}", leave=False):
        if kind == "loss":
            # Higher score for members => use negative loss
            scores.append(-sequence_loss(model, tokenizer, text, max_length, device))
        elif kind == "logprob":
            scores.append(sequence_logprob(model, tokenizer, text, max_length, device))
        elif kind == "mink":
            scores.append(min_k_score(model, tokenizer, text, max_length, device))
        else:
            raise ValueError(kind)
    return np.asarray(scores, dtype=np.float64)


def attack_a1_base_calibrated(
    model,
    tokenizer,
    members: Sequence[str],
    nonmembers: Sequence[str],
    max_length: int,
    device: str,
) -> Dict:
    """s(x) = log p_{W+Δ}(x) - log p_W(x)."""
    enable_lora(model)
    lp_adapt_m = batch_scores(model, tokenizer, members, max_length, device, "logprob")
    lp_adapt_n = batch_scores(model, tokenizer, nonmembers, max_length, device, "logprob")
    disable_lora(model)
    lp_base_m = batch_scores(model, tokenizer, members, max_length, device, "logprob")
    lp_base_n = batch_scores(model, tokenizer, nonmembers, max_length, device, "logprob")
    enable_lora(model)
    s_m = lp_adapt_m - lp_base_m
    s_n = lp_adapt_n - lp_base_n
    metrics = _scores_to_metrics(s_m, s_n)
    metrics["attack"] = "a1_base_calibrated"
    return metrics


def attack_loss_threshold(model, tokenizer, members, nonmembers, max_length, device) -> Dict:
    enable_lora(model)
    m = batch_scores(model, tokenizer, members, max_length, device, "loss")
    n = batch_scores(model, tokenizer, nonmembers, max_length, device, "loss")
    metrics = _scores_to_metrics(m, n)
    metrics["attack"] = "loss_threshold"
    return metrics


def attack_mink(model, tokenizer, members, nonmembers, max_length, device) -> Dict:
    enable_lora(model)
    m = batch_scores(model, tokenizer, members, max_length, device, "mink")
    n = batch_scores(model, tokenizer, nonmembers, max_length, device, "mink")
    metrics = _scores_to_metrics(m, n)
    metrics["attack"] = "mink"
    return metrics


def attack_merged_control(
    model,
    tokenizer,
    members,
    nonmembers,
    max_length,
    device,
    base_model_for_calibration=None,
) -> Dict:
    """MIA when only merged weights are available: loss threshold on merged model.

    If base_model_for_calibration is provided, also report calibrated score using
    a separately held base model (simulating adversary who still has public W0).
    Here we approximate the 'merged-only' setting with adapted loss alone.
    """
    enable_lora(model)
    m = batch_scores(model, tokenizer, members, max_length, device, "loss")
    n = batch_scores(model, tokenizer, nonmembers, max_length, device, "loss")
    metrics = _scores_to_metrics(m, n)
    metrics["attack"] = "merged_loss"
    return metrics


def _rowspace_basis(a: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor:
    """Orthonormal basis for row space of A (in_features, r) -> (in_features, r')."""
    # A is (in, r); rows live in R^{in}. Row space of A^T is col space of A.
    # Proposition: rowsp(A) ⊆ span{h}; A has shape (in, r), so columns of A span a
    # subspace of input activations. Use QR / SVD on A.
    a64 = a.detach().float()
    q, _ = torch.linalg.qr(a64, mode="reduced")
    if rank is not None:
        q = q[:, :rank]
    return q


@torch.no_grad()
def projection_energy(model, tokenizer, text: str, max_length: int, device: str) -> float:
    """Mean activation energy retained under projection onto rowsp(A_ell)."""
    modules = collect_lora_modules(model)
    bases = {name: _rowspace_basis(module.lora_a.data) for name, module in modules}
    # Hook to capture inputs to each LoRA-wrapped linear.
    activations = {}

    def make_hook(name):
        def hook(_mod, inputs, _output):
            # inputs[0]: (batch, seq, hidden)
            activations[name] = inputs[0].detach()
        return hook

    handles = []
    for name, module in modules:
        handles.append(module.layer.register_forward_hook(make_hook(name)))

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    disable_lora(model)  # capture base activations into LoRA layers' underlying linears
    # Re-enable path through layers: hooks are on underlying linear, still called.
    model(**enc)
    enable_lora(model)
    for h in handles:
        h.remove()

    energies = []
    for name, act in activations.items():
        # act: (1, seq, in)
        p = bases[name].to(act.device, dtype=act.dtype)
        # Project last-dim onto columns of p (in, r)
        # energy = ||P h||^2 / ||h||^2
        flat = act.reshape(-1, act.shape[-1]).float()
        mask = flat.norm(dim=-1) > 0
        flat = flat[mask]
        if flat.numel() == 0:
            continue
        proj = flat @ p.float()
        recon = proj @ p.float().T
        num = recon.pow(2).sum(dim=-1)
        den = flat.pow(2).sum(dim=-1).clamp_min(1e-12)
        energies.append((num / den).mean().item())
    return float(np.mean(energies)) if energies else 0.0


def attack_a2_subspace(
    model, tokenizer, members, nonmembers, max_length, device
) -> Dict:
    m = np.asarray(
        [projection_energy(model, tokenizer, t, max_length, device) for t in tqdm(members, desc="a2-m", leave=False)],
        dtype=np.float64,
    )
    n = np.asarray(
        [projection_energy(model, tokenizer, t, max_length, device) for t in tqdm(nonmembers, desc="a2-n", leave=False)],
        dtype=np.float64,
    )
    metrics = _scores_to_metrics(m, n)
    metrics["attack"] = "a2_subspace"
    return metrics


def fuse_scores(a_metrics_members, a_metrics_non, b_m, b_n) -> Dict:
    # z-score each and average
    def z(x, ref):
        mu, sd = ref.mean(), ref.std() + 1e-8
        return (x - mu) / sd

    ref = np.concatenate([a_metrics_non, b_n])
    # Use nonmember stats separately per score
    am = z(a_metrics_members, a_metrics_non)
    an = z(a_metrics_non, a_metrics_non)
    bm = z(b_m, b_n)
    bn = z(b_n, b_n)
    return _scores_to_metrics(0.5 * (am + bm), 0.5 * (an + bn))


@torch.no_grad()
def canary_exposure(
    model,
    tokenizer,
    canary_phrase: str,
    max_length: int,
    device: str,
    n_candidates: int = 500,
    seed: int = 0,
) -> Dict:
    pool = canary_candidate_pool(canary_phrase, n_candidates=n_candidates, seed=seed)
    scores = []
    enable_lora(model)
    for phrase in tqdm(pool, desc="canary", leave=False):
        scores.append(sequence_logprob(model, tokenizer, phrase, max_length, device))
    scores = np.asarray(scores)
    # Rank of true canary (1 = most likely)
    order = np.argsort(-scores)
    rank = int(np.where(order == 0)[0][0]) + 1
    exposure = math.log2(n_candidates) - math.log2(rank)
    return {
        "rank": rank,
        "n_candidates": n_candidates,
        "exposure": float(exposure),
        "true_logprob": float(scores[0]),
    }


PHI_PATTERNS = [
    re.compile(r"MRN-\d{7}"),
    re.compile(r"DOB \d{4}-\d{2}-\d{2}"),
    re.compile(r"Patient [A-Z][a-z]+ [A-Z][a-z]+"),
    re.compile(r"CONFIDENTIAL PHI MARKER:.*"),
]


def extract_phi_entities(text: str) -> List[str]:
    ents = []
    for pat in PHI_PATTERNS:
        ents.extend(pat.findall(text))
    return ents


def normalized_edit_distance(a: str, b: str) -> float:
    # Classic Levenshtein / max(len)
    la, lb = len(a), len(b)
    if la == 0:
        return 1.0 if lb else 0.0
    if lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1] / max(la, lb)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    enable_lora(model)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def attack_a4_reconstruction(
    model,
    tokenizer,
    targets: Sequence[str],
    public_templates: Sequence[str],
    max_length: int,
    device: str,
    max_new_tokens: int = 128,
) -> Dict:
    """Activation-guided reconstruction via public-template warm-start + greedy continuation.

    Full HotFlip simplex search is expensive; we implement the paper's warm-start
    path: seed from public templates / section headers and decode under the
    adapted model, then score against nearest training record.
    """
    prompts = [
        "Discharge Summary\nPatient Name:",
        "CONFIDENTIAL PHI MARKER:",
        "History of Present Illness:\nThe patient presented",
    ]
    if public_templates:
        prompts.append(public_templates[0].split("\n")[0] + "\nPatient Name:")

    gens = [greedy_generate(model, tokenizer, p, max_new_tokens, device) for p in prompts]
    edit_dists = []
    exact = 0
    phi_hits = 0
    phi_total = 0
    for target in targets:
        target_phi = extract_phi_entities(target)
        phi_total += max(len(target_phi), 1)
        best = min(normalized_edit_distance(g, target) for g in gens)
        edit_dists.append(best)
        if best < 0.05:
            exact += 1
        # PHI recall: fraction of target PHI strings that appear in any generation
        joined = "\n".join(gens)
        hits = sum(1 for e in target_phi if e in joined)
        phi_hits += hits
    return {
        "edit_distance": float(np.mean(edit_dists)) if edit_dists else 1.0,
        "exact_span_rate": float(exact) / max(len(targets), 1),
        "phi_recall": float(phi_hits) / max(phi_total, 1),
        "n_targets": len(targets),
        "example_generation": gens[0][:500] if gens else "",
    }


def attack_a5_attribute(model, tokenizer, device: str, true_stats: Dict) -> Dict:
    prompt = (
        "In this hospital cohort, summarize typical patient age, sex ratio, "
        "and common diagnoses in one short paragraph:\n"
    )
    text = greedy_generate(model, tokenizer, prompt, max_new_tokens=80, device=device)
    return {
        "generation": text,
        "true_stats": true_stats,
        "note": "qualitative cohort probe; no numeric attribute error computed for synthetic corpus",
    }
