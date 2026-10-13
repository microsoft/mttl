"""Diagnostics for the two remaining theory todos in NAACL_privacy_lora.tex.

Todo 1 (Table 1 paragraph):
  * R_exact — rebuild B_T from logged per-example SGD grads (Thm. exact superposition)
  * eps_A, eps_G — drift quantities in Prop. drift bound; compare ||E_T|| to the bound
  * subset-vs-union additivity — ||ΔW_union − Σ ΔW_parts|| / ||ΔW_union||

Todo 2 (Table 2 paragraph):
  * dictionary coherence μ and fraction of runs with κ=1-(2s-1)μ > 0
  * perturbation sweep at 1%, 5%, 10%, 20%
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

from projects.privacy_lora.v2.superposition import (
    DATASETS,
    LR,
    M,
    MODELS,
    S,
    SEEDS,
    _best_subset,
    _e_c_train,
    _features,
    _module,
    _prepare,
    _set_ab,
    _support_hit,
    _texts,
    _train_weights,
    attach,
)
from projects.privacy_lora.v2.modeling import load_base_model

OUT = os.environ.get("PRIVACY_LORA_TODOS", "privacy_lora_v2/superposition_todos")
TAU = 1e-12
PERTURB_LEVELS = (0.01, 0.05, 0.10, 0.20)
# Horizons for identity/drift; additivity at short + mid.
DIAG_STEPS = (1, 50, 500)
ADD_STEPS = (1, 50)


def _opnorm(A: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(A.float().cpu(), ord=2).item())


def _fro(A: torch.Tensor) -> float:
    return float(torch.linalg.norm(A.float().cpu(), "fro").item())


def _loss_G(model, module, tok, enc, device: str):
    """Per-example loss, ∇_B L, and effective G=∂L/∂ΔW for ΔW = scaling·A@B."""
    ids, n_prompt = enc
    x = torch.tensor([ids], device=device)
    attn = torch.ones_like(x)
    labels = torch.full_like(x, -100)
    labels[0, n_prompt:] = x[0, n_prompt:]
    cache: Dict[str, torch.Tensor] = {}

    def fwd(_m, inp, _out):
        cache["h"] = inp[0].detach()

    def bwd(_m, _gin, gout):
        cache["delta"] = gout[0].detach()

    h1 = module.register_forward_hook(fwd)
    h2 = module.register_full_backward_hook(bwd)
    out = model(input_ids=x, attention_mask=attn, labels=labels)
    loss = out.loss
    loss.backward()
    gB = module.lora_b.grad.detach().float().clone() if module.lora_b.grad is not None else None
    gA = module.lora_a.grad.detach().float().clone() if module.lora_a.grad is not None else None
    h = cache["h"].float().reshape(-1, cache["h"].shape[-1])
    delta = cache["delta"].float().reshape(-1, cache["delta"].shape[-1])
    # For right-multiply ΔW (out = h @ ΔW): G = ∂L/∂ΔW = h^T δ.
    G = h.transpose(0, 1) @ delta  # (d_in, d_out)
    model.zero_grad(set_to_none=True)
    h1.remove()
    h2.remove()
    return loss.detach(), gB, gA, G


def train_instrumented(model, module, tok, train_encs, steps: int, lr: float, freeze_a: bool, device: str):
    """Clean SGD while logging path quantities needed for R_exact / ε_A / ε_G."""
    module.lora_a.requires_grad_(not freeze_a)
    module.lora_b.requires_grad_(True)
    A0 = module.lora_a.detach().float().clone()
    # G_0 at initialization (B=0).
    _set_ab(module, module.lora_a.data, torch.zeros_like(module.lora_b))
    G0 = []
    for enc in train_encs:
        model.zero_grad(set_to_none=True)
        _, _, _, G = _loss_G(model, module, tok, enc, device)
        G0.append(G.cpu())
    _set_ab(module, A0.to(module.lora_a.dtype), torch.zeros_like(module.lora_b))

    B_recon = torch.zeros_like(module.lora_b, dtype=torch.float32).cpu()
    A_recon = A0.detach().float().cpu().clone()
    eps_A = 0.0
    eps_G = 0.0
    a_max = _opnorm(A0)
    g0_max = max(_fro(g) for g in G0)

    model.train()
    for _t in range(steps):
        A_t = module.lora_a.detach().float().clone()
        eps_A = max(eps_A, _opnorm(A_t.cpu() - A0.cpu()))
        a_max = max(a_max, _opnorm(A_t))

        gB_list = []
        gA_list = []
        for j, enc in enumerate(train_encs):
            model.zero_grad(set_to_none=True)
            _, gB, gA, G = _loss_G(model, module, tok, enc, device)
            if gB is None:
                raise RuntimeError("missing B grad")
            gB_list.append(gB)
            if not freeze_a and gA is not None:
                gA_list.append(gA)
            eps_G = max(eps_G, _fro(G.cpu() - G0[j]))

        gB_mean = torch.stack(gB_list, 0).mean(0)
        B_recon = B_recon - lr * gB_mean.detach().float().cpu()
        with torch.no_grad():
            module.lora_b.data.add_(-lr * gB_mean.to(module.lora_b.dtype))
            if not freeze_a and gA_list:
                gA_mean = torch.stack(gA_list, 0).mean(0)
                A_recon = A_recon - lr * gA_mean.detach().float().cpu()
                module.lora_a.data.add_(-lr * gA_mean.to(module.lora_a.dtype))

    model.eval()
    BT = module.lora_b.detach().float().cpu()
    AT = module.lora_a.detach().float().cpu()
    r_exact = float((BT - B_recon).norm() / max(BT.norm().item(), TAU))
    # Fixed-feature residual in B-space using G_0: Φ_i ∝ ∇_B at init.
    return {
        "R_exact": r_exact,
        "eps_A": eps_A,
        "eps_G": eps_G,
        "a_max": a_max,
        "a0": _opnorm(A0),
        "g0_max": g0_max,
        "AT": AT,
        "BT": BT,
        "A0": A0.cpu(),
        "G0": G0,
    }


def _delta_w(A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
    return scaling * (A.float() @ B.float())


def drift_and_bound(info: dict, Phi0_train: torch.Tensor, steps: int, scaling: float, s: int):
    """||E_T||_F vs Prop. bound; E_T = ΔW_T − Σ_i c_i Φ_i^W with c_i=ηT/s on support."""
    A0, AT, BT = info["A0"], info["AT"], info["BT"]
    c = abs(_train_weights(steps, LR, s, scaling))  # positive weight magnitude
    # Φ^B from features (= ∇_B/scaling); fixed-feature B_ff = −(ηT/s)·scaling · Σ Φ = −c_signed Σ Φ
    # Our c_star = −ηT scaling / s; B_ff = Σ c_star Φ_i = c_star * Σ Φ_i
    c_star = _train_weights(steps, LR, s, scaling)
    B_ff = c_star * Phi0_train.float().cpu().sum(0)
    dW_T = _delta_w(AT, BT, scaling)
    dW_ff = _delta_w(A0, B_ff, scaling)
    # Prefer comparing in B-space mapped by final A for a fair ΔW residual when A drifted:
    dW_ff_AT = _delta_w(AT, B_ff, scaling)
    E = dW_T - dW_ff_AT
    e_norm = _fro(E)
    dw_norm = max(_fro(dW_T), TAU)
    gamma = scaling  # α/r; with α=r this is 1
    Lam = LR * steps
    a, a0, g, eA, eG = info["a_max"], info["a0"], info["g0_max"], info["eps_A"], info["eps_G"]
    bound = (gamma ** 2) * Lam * (a * a * eG + g * (a + a0) * eA)
    return {
        "E_T_rel": float(e_norm / dw_norm),
        "E_T_fro": e_norm,
        "bound_rhs": float(bound),
        "bound_ok": bool(e_norm <= bound + 1e-6),
        "R_ff_cstar": float((BT - B_ff).norm() / max(BT.norm().item(), TAU)),
    }


def coherence_stats(Phi: torch.Tensor, s: int = S) -> Dict[str, float]:
    V = Phi.reshape(Phi.shape[0], -1).float().cpu()
    norms = V.norm(dim=1).clamp_min(1e-12)
    U = V / norms[:, None]
    G = U @ U.T
    Mloc = G.shape[0]
    off = []
    for i in range(Mloc):
        for j in range(i + 1, Mloc):
            off.append(abs(float(G[i, j])))
    mu = float(max(off)) if off else 0.0
    kappa = float(1.0 - (2 * s - 1) * mu)
    return {"mu": mu, "kappa": kappa, "kappa_pos": 1.0 if kappa > 0 else 0.0}


def perturb_sweep(Phi0: torch.Tensor, BT: torch.Tensor, truth, c_star: float) -> Dict[str, dict]:
    out = {}
    for pct in PERTURB_LEVELS:
        noise = torch.randn_like(BT)
        Bpert = BT + pct * BT.float().norm() * noise / (noise.norm() + 1e-12)
        sub, ch, rf = _best_subset(Phi0, Bpert, S)
        out[f"perturb_{int(pct*100)}pct"] = {
            "E_c": _e_c_train(ch, sub, c_star, truth, M),
            "supp": 100.0 if _support_hit(sub, truth) else 0.0,
            "R_fit": rf,
        }
    return out


def train_plain(model, module, tok, train_encs, steps: int, lr: float, freeze_a: bool, device: str):
    module.lora_a.requires_grad_(not freeze_a)
    module.lora_b.requires_grad_(True)
    model.train()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        losses = []
        for enc in train_encs:
            ids, n_prompt = enc
            x = torch.tensor([ids], device=device)
            labels = torch.full_like(x, -100)
            labels[0, n_prompt:] = x[0, n_prompt:]
            losses.append(model(input_ids=x, attention_mask=torch.ones_like(x), labels=labels).loss)
        torch.stack(losses).mean().backward()
        with torch.no_grad():
            if module.lora_b.grad is not None:
                module.lora_b.data.add_(-lr * module.lora_b.grad)
            if (not freeze_a) and module.lora_a.grad is not None:
                module.lora_a.data.add_(-lr * module.lora_a.grad)
    model.eval()


def additivity(model_key, name, layer, tok_encs_builder, truth, A0_template, steps, device):
    """Train on union vs sum of singleton adapters from the same A0."""
    # Fresh model helper
    def run_on(subset: Sequence[int]):
        model, tok = load_base_model(name, device)
        attach(model, layer, 16, False)
        module = _module(model)
        module.lora_a.data.copy_(A0_template.to(module.lora_a.dtype))
        module.lora_b.data.zero_()
        encs = [tok_encs_builder[i] for i in subset]
        train_plain(model, module, tok, encs, steps, LR, False, device)
        A = module.lora_a.detach().float().cpu()
        B = module.lora_b.detach().float().cpu()
        sc = float(module.scaling)
        del model
        torch.cuda.empty_cache()
        return A, B, sc

    AU, BU, sc = run_on(truth)
    dW_U = _delta_w(AU, BU, sc)
    dW_sum = torch.zeros_like(dW_U)
    for i in truth:
        Ai, Bi, sci = run_on([i])
        dW_sum = dW_sum + _delta_w(Ai, Bi, sci)
    # Mean-batch union uses 1/s per example; singletons use weight 1. Scale-match before comparing.
    s = max(len(truth), 1)
    dW_sum_matched = dW_sum / s
    rel_raw = float(torch.linalg.norm(dW_U - dW_sum, "fro") / max(torch.linalg.norm(dW_U, "fro").item(), TAU))
    rel = float(torch.linalg.norm(dW_U - dW_sum_matched, "fro") / max(torch.linalg.norm(dW_U, "fro").item(), TAU))
    return {"R_add": rel, "R_add_raw": rel_raw, "steps": steps, "n_parts": s}


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
        "identity_drift": {},
        "additivity": {},
        "coherence": {},
        "perturbation": {},
    }

    # Shared A0 for r=16
    attach(model, layer, 16, False)
    module = _module(model)
    A0 = module.lora_a.detach().clone()
    scaling = float(module.scaling)

    for steps in DIAG_STEPS:
        # reset
        del model
        torch.cuda.empty_cache()
        model, tok = load_base_model(name, device)
        attach(model, layer, 16, False)
        module = _module(model)
        module.lora_a.data.copy_(A0)
        module.lora_b.data.zero_()

        info = train_instrumented(model, module, tok, train_encs, steps, LR, False, device)
        # Features at A0 for c*-residual / coherence / perturb (reuse after reset B)
        _set_ab(module, A0, torch.zeros_like(module.lora_b))
        Phi0 = _features(model, module, tok, encs, A0, device)
        Phi_train = Phi0[list(truth)]
        drift = drift_and_bound(info, Phi_train, steps, scaling, S)
        rec = {
            "R_exact": info["R_exact"],
            "eps_A": info["eps_A"],
            "eps_G": info["eps_G"],
            **drift,
        }
        out["identity_drift"][f"t{steps}"] = rec
        print(
            f"[{model_key} {dataset} s{seed} t{steps}] "
            f"R_exact={rec['R_exact']:.2e} eps_A={rec['eps_A']:.3f} eps_G={rec['eps_G']:.3f} "
            f"E_T_rel={rec['E_T_rel']:.3f} bound_ok={rec['bound_ok']}",
            flush=True,
        )

        if steps == 50:
            coh = coherence_stats(Phi0, S)
            out["coherence"] = coh
            c_star = _train_weights(steps, LR, S, scaling)
            # restore BT into module for perturb (already have info['BT'])
            BT = info["BT"].to(module.lora_b.device)
            out["perturbation"] = perturb_sweep(Phi0, BT, truth, c_star)
            print(
                f"[{model_key} {dataset} s{seed} coh] mu={coh['mu']:.4f} kappa={coh['kappa']:.4f} "
                f"kappa_pos={coh['kappa_pos']:.0f}",
                flush=True,
            )
            for k, v in out["perturbation"].items():
                print(f"  {k}: E_c={v['E_c']:.3f} supp={v['supp']:.0f} R_fit={v['R_fit']:.3f}", flush=True)

    # Additivity controls (separate trains)
    for steps in ADD_STEPS:
        rel = additivity(model_key, name, layer, encs, truth, A0, steps, device)
        out["additivity"][f"t{steps}"] = rel
        print(f"[{model_key} {dataset} s{seed} add t{steps}] R_add={rel['R_add']:.3f}", flush=True)

    path = os.path.join(OUT, f"{model_key}_{dataset}_s{seed}.json")
    os.makedirs(OUT, exist_ok=True)
    # drop non-serializable
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="qwen14b")
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
