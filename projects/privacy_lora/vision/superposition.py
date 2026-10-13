"""Clean-SGD LoRA superposition / support recovery on MedMNIST (vision).

Mirrors the NLP protocol in projects.privacy_lora.v2.superposition:
  * one middle-block attention out-projection with LoRA
  * M=20 candidates, s=4 hidden participants
  * α=r; clean SGD; steps in {1,10,50,200,500}
  * E_c vs training-implied weights c^* = −η T / s
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

from projects.privacy_lora.vision.backbone import MiniViT
from projects.privacy_lora.vision.data import DATASETS, sample_pool

OUT = os.environ.get("PRIVACY_LORA_VISION", "privacy_lora_v2/vision_superposition")
M, S = 20, 4
LR = 5e-3
SEEDS = (0, 1, 2)
IMG = 64
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


def _ls(dict_cols: torch.Tensor, target: torch.Tensor, idx: Sequence[int]) -> Tuple[torch.Tensor, float]:
    D = dict_cols[list(idx)].reshape(len(idx), -1).T.detach().double().cpu()
    y = target.reshape(-1).detach().double().cpu()
    if not torch.isfinite(D).all() or not torch.isfinite(y).all():
        return torch.zeros(len(idx)), 1.0
    sol = torch.linalg.lstsq(D, y, rcond=None).solution
    pred = D @ sol
    rel = float((y - pred).norm() / (y.norm() + 1e-12))
    return sol.float(), rel


def _best_subset(dict_cols: torch.Tensor, target: torch.Tensor, s: int = S):
    best = None
    for comb in itertools.combinations(range(dict_cols.shape[0]), s):
        c, rel = _ls(dict_cols, target, comb)
        if best is None or rel < best[2]:
            best = (comb, c, rel)
    return best


def _dense_ls(dict_cols: torch.Tensor, target: torch.Tensor):
    D = dict_cols.reshape(dict_cols.shape[0], -1).T.detach().double().cpu()
    y = target.reshape(-1).detach().double().cpu()
    sol = torch.linalg.lstsq(D, y, rcond=None).solution.float()
    pred = D.float() @ sol
    rel = float((y.float() - pred).norm() / (y.float().norm() + 1e-12))
    return sol, rel


def _support_hit(est, truth):
    return set(est) == set(truth)


def _train_weights(steps, lr, s, scaling):
    return float(-lr * steps * scaling / s)


def _e_c_train(c_hat, idx_hat, c_star, idx_true, m):
    a = torch.zeros(m)
    b = torch.zeros(m)
    a[list(idx_hat)] = c_hat.detach().cpu().float()
    b[list(idx_true)] = float(c_star)
    return float((a - b).norm() / (b.norm() + 1e-12))


def _set_ab(module, A, B):
    module.lora_a.data.copy_(A.to(module.lora_a.device, module.lora_a.dtype))
    module.lora_b.data.copy_(B.to(module.lora_b.device, module.lora_b.dtype))


def _loss_feat(model, module, x, y, want_feat: bool):
    model.zero_grad(set_to_none=True)
    out = model(x.unsqueeze(0), labels=y.unsqueeze(0) if y.ndim == 0 else y.view(1))
    loss = out["loss"]
    feat = None
    if want_feat:
        loss.backward()
        feat = module.lora_b.grad.detach().float().clone() if module.lora_b.grad is not None else None
        model.zero_grad(set_to_none=True)
    return loss, feat


def _features(model, module, images, labels, A, device):
    B0 = torch.zeros_like(module.lora_b)
    _set_ab(module, A, B0)
    model.eval()
    cols = []
    for i in range(images.shape[0]):
        x = images[i].to(device)
        y = labels[i].to(device)
        _, feat = _loss_feat(model, module, x, y, want_feat=True)
        if feat is None:
            raise RuntimeError("missing B-grad")
        cols.append(feat / max(float(module.scaling), 1e-8))
    return torch.stack(cols, 0)


@torch.no_grad()
def _nll(model, x, y):
    return float(model(x.unsqueeze(0), labels=y.view(1))["loss"].item())


def _readout_mae(model, module, images, labels, A_keep, B_true, B_hat, device):
    errs = []
    for i in range(images.shape[0]):
        x, y = images[i].to(device), labels[i].to(device)
        _set_ab(module, A_keep, B_true)
        t = _nll(model, x, y)
        _set_ab(module, A_keep, B_hat)
        h = _nll(model, x, y)
        errs.append(abs(h - t))
    return float(np.mean(errs))


def train_clean(model, module, images, labels, idx, steps, lr, freeze_a, device):
    module.lora_a.requires_grad_(not freeze_a)
    module.lora_b.requires_grad_(True)
    model.train()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        losses = []
        for i in idx:
            out = model(images[i].unsqueeze(0).to(device), labels=labels[i].view(1).to(device))
            losses.append(out["loss"])
        torch.stack(losses).mean().backward()
        with torch.no_grad():
            if module.lora_b.grad is not None:
                module.lora_b.data.add_(-lr * module.lora_b.grad)
            if (not freeze_a) and module.lora_a.grad is not None:
                module.lora_a.data.add_(-lr * module.lora_a.grad)
    model.eval()


def make_model(n_classes: int, rank: int, freeze_a: bool, device: str):
    model = MiniViT(
        num_classes=n_classes, img_size=IMG, patch=8, dim=256, depth=6, heads=4,
        lora_layer=3, rank=rank, freeze_a=freeze_a, in_ch=3,
    ).to(device)
    return model, model.lora_module()


def run_group(dataset: str, seed: int, device: str = "cuda") -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    pool = sample_pool(dataset, M, seed, img_size=IMG)
    images, labels = pool["images"], pool["labels"]
    truth = tuple(sorted(random.Random(seed + 17).sample(range(M), S)))
    n_classes = pool["n_classes"]

    out = {
        "modality": "vision",
        "dataset": dataset,
        "clinical": pool["clinical"],
        "seed": seed,
        "truth": list(truth),
        "M": M,
        "S": S,
        "lr": LR,
        "backbone": "MiniViT-256-d6",
        "table1": {},
        "table2": {},
    }
    snapshots = {}

    for cfg in TABLE1:
        model, module = make_model(n_classes, cfg["rank"], cfg["freeze_a"], device)
        key = cfg["rank"]
        if key not in snapshots:
            snapshots[key] = module.lora_a.detach().clone()
        else:
            module.lora_a.data.copy_(snapshots[key])
            module.lora_b.data.zero_()
        A0 = module.lora_a.detach().clone()
        train_clean(model, module, images, labels, truth, cfg["steps"], LR, cfg["freeze_a"], device)
        AT = module.lora_a.detach().clone()
        BT = module.lora_b.detach().clone()
        scaling = float(module.scaling)
        c_star = _train_weights(cfg["steps"], LR, S, scaling)

        Phi0 = _features(model, module, images, labels, A0, device)
        _, r_ff = _ls(Phi0, BT, truth)
        subset, c_hat, r_fit = _best_subset(Phi0, BT, S)
        B_hat = (c_hat.view(S, 1, 1).to(Phi0.device) * Phi0[list(subset)]).sum(0)
        e_read = _readout_mae(model, module, images, labels, AT, BT, B_hat, device)
        rec = {
            "R_ff": r_ff,
            "supp": 100.0 if _support_hit(subset, truth) else 0.0,
            "E_read": e_read,
            "R_fit": r_fit,
            "subset": list(subset),
            "E_c": _e_c_train(c_hat, subset, c_star, truth, M),
            "c_star": c_star,
        }
        out["table1"][cfg["label"]] = rec

        if cfg["label"] == ABLATION_LABEL:
            def pack(sub, ch, rf):
                return {
                    "E_c": _e_c_train(ch, sub, c_star, truth, M),
                    "supp": 100.0 if _support_hit(sub, truth) else 0.0,
                    "R_fit": rf,
                }

            abl = {"known_A0_sparse": {"E_c": rec["E_c"], "supp": rec["supp"], "R_fit": rec["R_fit"]}}
            PhiT = _features(model, module, images, labels, AT, device)
            sub, ch, rf = _best_subset(PhiT, BT, S)
            abl["A_T_proxy"] = pack(sub, ch, rf)
            Aind = torch.randn_like(A0) * A0.std().clamp_min(1e-4)
            PhiI = _features(model, module, images, labels, Aind, device)
            sub, ch, rf = _best_subset(PhiI, BT, S)
            abl["indep_init"] = pack(sub, ch, rf)
            c_all, rf = _dense_ls(Phi0, BT)
            top = tuple(sorted(torch.topk(c_all.abs(), S).indices.tolist()))
            abl["no_sparsity"] = pack(top, c_all[list(top)], rf)
            noise = torch.randn_like(BT)
            Bpert = BT + 0.01 * BT.float().norm() * noise / (noise.norm() + 1e-12)
            sub, ch, rf = _best_subset(Phi0, Bpert, S)
            abl["perturb_1pct"] = pack(sub, ch, rf)
            out["table2"] = abl

        print(
            f"[{dataset} s{seed} {cfg['label']}] "
            f"R_ff={rec['R_ff']:.3f} supp={rec['supp']:.0f} E_read={rec['E_read']:.3f} E_c={rec['E_c']:.3f}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    path = os.path.join(OUT, f"{dataset}_s{seed}.json")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for seed in seeds:
        dest = os.path.join(OUT, f"{args.dataset}_s{seed}.json")
        if os.path.exists(dest):
            print("[skip]", dest)
            continue
        run_group(args.dataset, seed)


if __name__ == "__main__":
    main()
