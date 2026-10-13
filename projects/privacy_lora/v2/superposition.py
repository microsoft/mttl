"""Clean-SGD superposition / support-recovery grid for Tables 1–2.

Hardened protocol (v3):
  * Qwen2.5-14B-Instruct primary; optional Llama baselines retained in MODELS
  * one middle-block attention o-projection
  * M=20 candidates, s=4 hidden training participants  (was 12/3)
  * steps include long horizons (1…500) so fixed-feature recovery can fail
  * α=r; clean SGD (no momentum, decay, dropout, clip, or DP)
  * E_c vs training-implied weights c^* = −η T / s (not tautological LS-vs-LS)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from projects.privacy_lora.v2.attacks import encode_example
from projects.privacy_lora.v2.data import CorpusSpec, Example, get_corpus
from projects.privacy_lora.v2.modeling import attach_method, load_base_model, lora_modules

OUT = os.environ.get("PRIVACY_LORA_SUPERPOS", "privacy_lora_v2/superposition_v3")
M, S = 20, 4
MAX_LEN = 256
LR = 5e-3
SEEDS = (0, 1, 2)

MODELS = {
    "llama1b": ("meta-llama/Llama-3.2-1B", 8),
    "llama8b": ("meta-llama/Llama-3.1-8B-Instruct", 16),
    "qwen14b": ("Qwen/Qwen2.5-14B-Instruct", 24),
}
DATASETS = ("pmc", "synthetic", "enron")
# Ablations attach to this Table-1 cell (same seed / corpus / A0).
ABLATION_LABEL = "sgd_r16_t50"
TABLE1 = [
    {"rank": 16, "steps": 1, "freeze_a": False, "label": "sgd_r16_t1"},
    {"rank": 16, "steps": 10, "freeze_a": False, "label": "sgd_r16_t10"},
    {"rank": 16, "steps": 50, "freeze_a": False, "label": "sgd_r16_t50"},
    {"rank": 16, "steps": 200, "freeze_a": False, "label": "sgd_r16_t200"},
    {"rank": 16, "steps": 500, "freeze_a": False, "label": "sgd_r16_t500"},
    {"rank": 1, "steps": 50, "freeze_a": False, "label": "sgd_r1_t50"},
    {"rank": 4, "steps": 50, "freeze_a": False, "label": "sgd_r4_t50"},
    {"rank": 16, "steps": 50, "freeze_a": True, "label": "frozenA_r16_t50"},
]


def _module(model):
    mods = lora_modules(model)
    if len(mods) != 1:
        raise RuntimeError(f"expected 1 LoRA module, got {len(mods)}: {[n for n, _ in mods]}")
    return mods[0][1]


def _texts(dataset: str, seed: int) -> List[str]:
    spec = CorpusSpec(dataset=dataset, n_train=120, n_holdout=40, n_val=10, n_public=10, n_public_shift=10, seed=seed)
    corpus = get_corpus(spec)
    notes = [t for t in corpus["members"] if len(t) >= 200][:M]
    if len(notes) < M:
        notes = (corpus["members"] + corpus["nonmembers"])[:M]
    if len(notes) < M:
        raise RuntimeError(f"{dataset}: only {len(notes)} notes")
    return notes


def _prepare(model, tok, texts: Sequence[str], device: str):
    return [encode_example(tok, Example("", t), MAX_LEN) for t in texts]


def _loss_and_features(model, module, tok, enc: Tuple[List[int], int], device: str, want_feat: bool):
    ids, n_prompt = enc
    x = torch.tensor([ids], device=device)
    attn = torch.ones_like(x)
    labels = torch.full_like(x, -100)
    labels[0, n_prompt:] = x[0, n_prompt:]
    cache = {}

    def fwd(_m, inp, _out):
        cache["h"] = inp[0].detach()

    h1 = module.register_forward_hook(fwd)
    out = model(input_ids=x, attention_mask=attn, labels=labels)
    loss = out.loss
    feat = None
    if want_feat:
        loss.backward(retain_graph=False)
        feat = module.lora_b.grad.detach().float().clone() if module.lora_b.grad is not None else None
        model.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)
    h1.remove()
    return loss, feat


@torch.no_grad()
def _nll(model, tok, enc, device: str) -> float:
    ids, n_prompt = enc
    x = torch.tensor([ids], device=device)
    labels = torch.full_like(x, -100)
    labels[0, n_prompt:] = x[0, n_prompt:]
    out = model(input_ids=x, attention_mask=torch.ones_like(x), labels=labels)
    return float(out.loss.item())


def _set_ab(module, A, B):
    module.lora_a.data.copy_(A.to(module.lora_a.device, module.lora_a.dtype))
    module.lora_b.data.copy_(B.to(module.lora_b.device, module.lora_b.dtype))


def _features(model, module, tok, encs, A, device: str) -> torch.Tensor:
    """Φ_j = ∇_B L(x_j) / scaling at the given A and B=0."""
    B0 = torch.zeros_like(module.lora_b)
    _set_ab(module, A, B0)
    model.eval()
    cols = []
    for enc in encs:
        model.zero_grad(set_to_none=True)
        _, feat = _loss_and_features(model, module, tok, enc, device, want_feat=True)
        if feat is None:
            raise RuntimeError("missing B-grad feature")
        cols.append(feat / max(float(module.scaling), 1e-8))
    model.zero_grad(set_to_none=True)
    return torch.stack(cols, dim=0)


def _ls(dict_cols: torch.Tensor, target: torch.Tensor, idx: Sequence[int]) -> Tuple[torch.Tensor, float]:
    # CPU double: avoids cusolver failures on large / ill-conditioned 14B features.
    D = dict_cols[list(idx)].reshape(len(idx), -1).T.detach().double().cpu()
    y = target.reshape(-1).detach().double().cpu()
    if not torch.isfinite(D).all() or not torch.isfinite(y).all():
        return torch.zeros(len(idx)), 1.0
    sol = torch.linalg.lstsq(D, y, rcond=None).solution
    pred = D @ sol
    rel = float((y - pred).norm() / (y.norm() + 1e-12))
    return sol.float(), rel


def _best_subset(dict_cols: torch.Tensor, target: torch.Tensor, s: int = S) -> Tuple[Tuple[int, ...], torch.Tensor, float]:
    best = None
    for comb in itertools.combinations(range(dict_cols.shape[0]), s):
        c, rel = _ls(dict_cols, target, comb)
        if best is None or rel < best[2]:
            best = (comb, c, rel)
    return best


def _dense_ls(dict_cols: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, float]:
    D = dict_cols.reshape(dict_cols.shape[0], -1).T.detach().double().cpu()
    y = target.reshape(-1).detach().double().cpu()
    if not torch.isfinite(D).all() or not torch.isfinite(y).all():
        return torch.zeros(dict_cols.shape[0]), 1.0
    sol = torch.linalg.lstsq(D, y, rcond=None).solution.float()
    pred = D.float() @ sol
    rel = float((y.float() - pred).norm() / (y.float().norm() + 1e-12))
    return sol, rel


def _best_subset_wspace(A0: torch.Tensor, Phi: torch.Tensor, BT: torch.Tensor, s: int = S):
    """Best-subset LS in ambient ΔW without materializing (d_in × d_out) tensors.

    Fitting A0@BT ≈ Σ c_i (A0@Φ_i) is equivalent (via H=A0ᵀA0) to fitting
    H^{1/2} BT ≈ Σ c_i H^{1/2} Φ_i in the LoRA factor space.
    """
    H = (A0.float().T @ A0.float()).double().cpu()
    evals, evecs = torch.linalg.eigh(H)
    Hs = (evecs * evals.clamp_min(0.0).sqrt()) @ evecs.T
    Hs = Hs.float()
    Phi_s = torch.einsum("ab,mbo->mao", Hs, Phi.float().cpu())
    tgt_s = Hs @ BT.float().cpu()
    return _best_subset(Phi_s, tgt_s, s)

def _support_hit(est: Sequence[int], truth: Sequence[int]) -> bool:
    return set(est) == set(truth)


def _train_weights(steps: int, lr: float, s: int, scaling: float) -> float:
    """Per-participant coefficient under fixed-Φ simultaneous SGD on mean loss.

    BT ≈ Σ_i c_i Φ_i with c_i = −η T · scaling / s for each training participant.
    With α=r, scaling is typically 1.
    """
    return float(-lr * steps * scaling / s)


def _e_c_train(c_hat: torch.Tensor, idx_hat: Sequence[int], c_star: float, idx_true: Sequence[int], m: int) -> float:
    a = torch.zeros(m)
    b = torch.zeros(m)
    a[list(idx_hat)] = c_hat.detach().cpu().float()
    b[list(idx_true)] = float(c_star)
    return float((a - b).norm() / (b.norm() + 1e-12))


def _readout_mae(model, module, tok, encs, A_keep, B_true, B_hat, device: str) -> float:
    errs = []
    for enc in encs:
        _set_ab(module, A_keep, B_true)
        nll_t = _nll(model, tok, enc, device)
        _set_ab(module, A_keep, B_hat)
        nll_h = _nll(model, tok, enc, device)
        errs.append(abs(nll_h - nll_t))
    return float(np.mean(errs))


def train_clean(model, module, tok, train_encs, steps: int, lr: float, freeze_a: bool, device: str):
    module.lora_a.requires_grad_(not freeze_a)
    module.lora_b.requires_grad_(True)
    params = [p for p in (module.lora_a, module.lora_b) if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for enc in train_encs:
            loss, _ = _loss_and_features(model, module, tok, enc, device, want_feat=False)
            losses.append(loss)
        torch.stack(losses).mean().backward()
        opt.step()
    model.eval()


def attach(model, layer: int, rank: int, freeze_a: bool):
    return attach_method(
        model, "lora", rank=rank, alpha=rank, targets="o_proj",
        modules_regex=rf".*layers\.{layer}\.self_attn", freeze_a=freeze_a, lora_dropout=0.0,
    )


def run_group(model_key: str, dataset: str, seed: int, device: str = "cuda") -> Dict:
    name, layer = MODELS[model_key]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    texts = _texts(dataset, seed)
    truth = tuple(sorted(random.Random(seed + 17).sample(range(M), S)))
    model, tok = load_base_model(name, device)
    encs = _prepare(model, tok, texts, device)
    train_encs = [encs[i] for i in truth]

    out = {
        "model": model_key,
        "dataset": dataset,
        "seed": seed,
        "truth": list(truth),
        "M": M,
        "S": S,
        "lr": LR,
        "table1": {},
        "table2": {},
    }
    snapshots = {}

    def fresh(rank: int, freeze_a: bool):
        nonlocal model, tok
        del model
        torch.cuda.empty_cache()
        model, tok = load_base_model(name, device)
        attach(model, layer, rank, freeze_a)
        return _module(model)

    for cfg in TABLE1:
        key = (cfg["rank"],)
        module = fresh(cfg["rank"], cfg["freeze_a"])
        if key not in snapshots:
            snapshots[key] = module.lora_a.detach().clone()
        else:
            module.lora_a.data.copy_(snapshots[key])
            module.lora_b.data.zero_()
        A0 = module.lora_a.detach().clone()
        train_clean(model, module, tok, train_encs, cfg["steps"], LR, cfg["freeze_a"], device)
        AT = module.lora_a.detach().clone()
        BT = module.lora_b.detach().clone()
        scaling = float(module.scaling)
        c_star = _train_weights(cfg["steps"], LR, S, scaling)

        Phi0 = _features(model, module, tok, encs, A0, device)
        _, r_ff = _ls(Phi0, BT, truth)
        subset, c_hat, r_fit = _best_subset(Phi0, BT, S)
        B_hat = (c_hat.view(S, 1, 1).to(Phi0.device) * Phi0[list(subset)]).sum(0)
        e_read = _readout_mae(model, module, tok, encs, AT, BT, B_hat, device)
        rec = {
            "R_ff": r_ff,
            "supp": 100.0 if _support_hit(subset, truth) else 0.0,
            "E_read": e_read,
            "R_fit": r_fit,
            "subset": list(subset),
            "E_c": _e_c_train(c_hat, subset, c_star, truth, M),
            "c_star": c_star,
            "scaling": scaling,
        }
        out["table1"][cfg["label"]] = rec

        if model_key == "qwen14b" and cfg["label"] == ABLATION_LABEL:
            def pack(sub, ch, rf):
                return {
                    "E_c": _e_c_train(ch, sub, c_star, truth, M),
                    "supp": 100.0 if _support_hit(sub, truth) else 0.0,
                    "R_fit": rf,
                }

            abl = {}
            abl["known_A0_sparse"] = {"E_c": rec["E_c"], "supp": rec["supp"], "R_fit": rec["R_fit"]}
            PhiT = _features(model, module, tok, encs, AT, device)
            sub, ch, rf = _best_subset(PhiT, BT, S)
            abl["A_T_proxy"] = pack(sub, ch, rf)
            Aind = torch.randn_like(A0) * A0.std().clamp_min(1e-4)
            PhiI = _features(model, module, tok, encs, Aind, device)
            sub, ch, rf = _best_subset(PhiI, BT, S)
            abl["indep_init"] = pack(sub, ch, rf)
            sub, ch, rf = _best_subset_wspace(A0, Phi0, BT, S)
            abl["no_lora_proj"] = pack(sub, ch, rf)
            c_all, rf = _dense_ls(Phi0, BT)
            top = tuple(sorted(torch.topk(c_all.abs(), S).indices.tolist()))
            abl["no_sparsity"] = pack(top, c_all[list(top)], rf)
            noise = torch.randn_like(BT)
            Bpert = BT + 0.01 * BT.float().norm() * noise / (noise.norm() + 1e-12)
            sub, ch, rf = _best_subset(Phi0, Bpert, S)
            abl["perturb_1pct"] = pack(sub, ch, rf)
            out["table2"] = abl

        print(
            f"[{model_key} {dataset} s{seed} {cfg['label']}] "
            f"R_ff={rec['R_ff']:.3f} supp={rec['supp']:.0f} E_read={rec['E_read']:.3f} E_c={rec['E_c']:.3f}",
            flush=True,
        )

    path = os.path.join(OUT, f"{model_key}_{dataset}_s{seed}.json")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--dataset", choices=DATASETS, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for ds in datasets:
        for seed in seeds:
            dest = os.path.join(OUT, f"{args.model}_{ds}_s{seed}.json")
            if os.path.exists(dest):
                print("[skip]", dest)
                continue
            run_group(args.model, ds, seed)


if __name__ == "__main__":
    main()
