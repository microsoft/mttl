"""Lightweight membership inference on MedMNIST LoRA adapters (vision audit).

Trains a MiniViT+LoRA on N member images, scores members vs holdout non-members
with loss and a base-calibrated loss (A1-style), reports AUC.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from projects.privacy_lora.vision.backbone import MiniViT
from projects.privacy_lora.vision.data import DATASETS, load_split, MedImages

OUT = os.environ.get("PRIVACY_LORA_VISION_MIA", "privacy_lora_v2/vision_mia")
IMG = 64
LR = 1e-3
EPOCHS = 5
N_MEM, N_NON = 200, 800


def _pack(imgs, labels, idxs, in_ch):
    ds = MedImages(imgs[idxs], labels[idxs], img_size=IMG, in_ch=3)
    xs, ys = [], []
    for i in range(len(idxs)):
        x, y = ds[i]
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def _losses(model, X, Y, device, batch=64):
    model.eval()
    out = []
    for i in range(0, len(Y), batch):
        x = X[i : i + batch].to(device)
        y = Y[i : i + batch].to(device)
        logits = model(x)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        out.append(loss.cpu())
    return torch.cat(out).numpy()


def run(dataset: str, seed: int, device: str = "cuda"):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    imgs, labels, n_classes, _ = load_split(dataset, "train")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(labels))
    mem_idx = perm[:N_MEM]
    non_idx = perm[N_MEM : N_MEM + N_NON]
    X_mem, Y_mem = _pack(imgs, labels, mem_idx, 3)
    X_non, Y_non = _pack(imgs, labels, non_idx, 3)

    model = MiniViT(num_classes=n_classes, img_size=IMG, lora_layer=3, rank=16).to(device)
    # unfreeze head for a usable classifier; LoRA already trainable
    for p in model.head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    for ep in range(EPOCHS):
        order = torch.randperm(N_MEM)
        total = 0.0
        for i in range(0, N_MEM, 32):
            b = order[i : i + 32]
            opt.zero_grad(set_to_none=True)
            loss = model(X_mem[b].to(device), labels=Y_mem[b].to(device))["loss"]
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(b)
        print(f"[{dataset} s{seed} ep{ep}] loss={total/N_MEM:.4f}", flush=True)

    # Base-calibrated score: loss(adapted) - loss(base-like with B=0)
    module = model.lora_module()
    B_keep = module.lora_b.detach().clone()
    loss_m = _losses(model, X_mem, Y_mem, device)
    loss_n = _losses(model, X_non, Y_non, device)
    with torch.no_grad():
        module.lora_b.zero_()
    base_m = _losses(model, X_mem, Y_mem, device)
    base_n = _losses(model, X_non, Y_non, device)
    with torch.no_grad():
        module.lora_b.copy_(B_keep)

    y = np.concatenate([np.ones(N_MEM), np.zeros(N_NON)])
    # higher score => more member-like; use negative loss
    auc_loss = float(roc_auc_score(y, -np.concatenate([loss_m, loss_n])))
    auc_a1 = float(roc_auc_score(y, -np.concatenate([loss_m - base_m, loss_n - base_n])))
    rec = {
        "dataset": dataset,
        "clinical": DATASETS[dataset][3],
        "seed": seed,
        "n_mem": N_MEM,
        "n_non": N_NON,
        "epochs": EPOCHS,
        "auc_loss": auc_loss,
        "auc_a1": auc_a1,
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{dataset}_s{seed}.json")
    json.dump(rec, open(path, "w"), indent=1)
    print("wrote", path, rec, flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dest = os.path.join(OUT, f"{args.dataset}_s{args.seed}.json")
    if os.path.exists(dest):
        print("[skip]", dest)
        return
    run(args.dataset, args.seed)


if __name__ == "__main__":
    main()
