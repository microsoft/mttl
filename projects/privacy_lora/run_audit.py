"""Run the LoRALeak experiment grid and write JSON results for the paper tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mttl.logging import logger, setup_logging
from projects.privacy_lora.attacks import (
    attack_a1_base_calibrated,
    attack_a2_subspace,
    attack_a4_reconstruction,
    attack_loss_threshold,
    attack_merged_control,
    attack_mink,
    canary_exposure,
    fuse_scores,
    batch_scores,
    _scores_to_metrics,
)
from projects.privacy_lora.clinical_data import build_corpus, load_corpus, save_corpus
from projects.privacy_lora.defenses import (
    defense_d2_truncate,
    defense_d4_merge_state_dicts,
    defense_d5_adaptguard,
)
from projects.privacy_lora.train_adapter import (
    TrainConfig,
    attach_lora,
    count_lora_params,
    load_adapted_model,
    load_base_model,
    load_lora_state,
    lora_state_dict,
    train_lora_on_notes,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="privacy_lora_artifacts")
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n_train", type=int, default=1000)
    p.add_argument("--n_holdout", type=int, default=500)
    p.add_argument("--n_public", type=int, default=300)
    p.add_argument("--n_mia", type=int, default=200, help="members/nonmembers scored for MIA")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", default="all",
                   choices=["data", "train", "mia", "rank", "recon", "defense", "all", "smoke"])
    p.add_argument("--ranks", default="1,2,4,8,16,64")
    p.add_argument("--base_rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--canary_candidates", type=int, default=256)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def corpus_path(out):
    return os.path.join(out, "corpus.json")


def adapter_dir(out, tag):
    return os.path.join(out, "adapters", tag)


def ensure_corpus(args):
    path = corpus_path(args.output_dir)
    if os.path.exists(path) and args.skip_existing:
        return load_corpus(path)
    corpus = build_corpus(
        n_train=args.n_train,
        n_holdout=args.n_holdout,
        n_public=args.n_public,
        seed=args.seed,
    )
    save_corpus(corpus, path)
    logger.info("Wrote corpus %s", path)
    return corpus


def train_one(args, texts, tag, **cfg_kw):
    out = adapter_dir(args.output_dir, tag)
    ckpt = os.path.join(out, "adapter.pt")
    if args.skip_existing and os.path.exists(ckpt):
        logger.info("Skip existing %s", ckpt)
        return ckpt
    cfg = TrainConfig(
        model_name=args.model_name,
        lora_rank=cfg_kw.get("lora_rank", args.base_rank),
        lora_alpha=float(cfg_kw.get("lora_rank", args.base_rank)),
        num_train_epochs=cfg_kw.get("epochs", args.epochs),
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        learning_rate=args.lr,
        seed=args.seed,
        optimizer=cfg_kw.get("optimizer", "adamw"),
        freeze_a=cfg_kw.get("freeze_a", False),
        dp_epsilon=cfg_kw.get("dp_epsilon"),
    )
    result = train_lora_on_notes(texts, cfg, out)
    return result["adapter_path"]


def eval_utility_proxy(model, tokenizer, texts, max_length, device, n=50):
    """Negative mean NLL as a utility proxy when no labeled task is available."""
    from projects.privacy_lora.attacks import sequence_loss

    losses = []
    for t in texts[:n]:
        losses.append(sequence_loss(model, tokenizer, t, max_length, device))
    # Map to a pseudo macro-F1 in [0,1] for the table: 1/(1+loss)
    mean_loss = float(np.mean(losses))
    return {"mean_nll": mean_loss, "utility_score": 1.0 / (1.0 + mean_loss)}


def run_mia(args, corpus, adapter_path, tag="mia"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg, meta = load_adapted_model(args.model_name, adapter_path, device)
    members = corpus["train"][: args.n_mia]
    nonmembers = corpus["holdout"][: args.n_mia]

    results = {}
    results["loss_threshold"] = attack_loss_threshold(
        model, tokenizer, members, nonmembers, args.max_length, device
    )
    results["mink"] = attack_mink(
        model, tokenizer, members, nonmembers, args.max_length, device
    )
    results["a1"] = attack_a1_base_calibrated(
        model, tokenizer, members, nonmembers, args.max_length, device
    )
    # A2 is slower; use a subset
    n_a2 = min(80, args.n_mia)
    results["a2"] = attack_a2_subspace(
        model, tokenizer, members[:n_a2], nonmembers[:n_a2], args.max_length, device
    )
    # Fusion: recompute raw scores cheaply via a1-style and a2 on subset
    # Approximate fusion by z-scoring reported isn't available; recompute:
    from projects.privacy_lora.attacks import sequence_logprob, projection_energy, disable_lora, enable_lora

    enable_lora(model)
    lp_a_m = np.array([sequence_logprob(model, tokenizer, t, args.max_length, device) for t in members[:n_a2]])
    lp_a_n = np.array([sequence_logprob(model, tokenizer, t, args.max_length, device) for t in nonmembers[:n_a2]])
    disable_lora(model)
    lp_b_m = np.array([sequence_logprob(model, tokenizer, t, args.max_length, device) for t in members[:n_a2]])
    lp_b_n = np.array([sequence_logprob(model, tokenizer, t, args.max_length, device) for t in nonmembers[:n_a2]])
    enable_lora(model)
    s_m = lp_a_m - lp_b_m
    s_n = lp_a_n - lp_b_n
    p_m = np.array([projection_energy(model, tokenizer, t, args.max_length, device) for t in members[:n_a2]])
    p_n = np.array([projection_energy(model, tokenizer, t, args.max_length, device) for t in nonmembers[:n_a2]])

    def z(x, ref):
        return (x - ref.mean()) / (ref.std() + 1e-8)

    fused_m = 0.5 * (z(s_m, s_n) + z(p_m, p_n))
    fused_n = 0.5 * (z(s_n, s_n) + z(p_n, p_n))
    results["a1_a2_fusion"] = _scores_to_metrics(fused_m, fused_n)
    results["a1_a2_fusion"]["attack"] = "a1_a2_fusion"

    results["merged_control"] = attack_merged_control(
        model, tokenizer, members, nonmembers, args.max_length, device
    )
    util = eval_utility_proxy(model, tokenizer, corpus["holdout"], args.max_length, device)
    results["utility"] = util
    results["lora_params"] = meta.get("lora_params")
    results["adapter"] = adapter_path

    out = os.path.join(args.output_dir, f"{tag}.json")
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Wrote %s", out)
    del model
    torch.cuda.empty_cache()
    return results


def run_rank_sweep(args, corpus):
    ranks = [int(x) for x in args.ranks.split(",") if x]
    rows = []
    for r in ranks:
        tag = f"rank_r{r}"
        path = train_one(args, corpus["train"], tag, lora_rank=r, epochs=args.epochs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, cfg, meta = load_adapted_model(args.model_name, path, device)
        members = corpus["train"][: args.n_mia]
        nonmembers = corpus["holdout"][: args.n_mia]
        a1 = attack_a1_base_calibrated(
            model, tokenizer, members, nonmembers, args.max_length, device
        )
        # canary with highest repetitions
        canaries = sorted(corpus["canaries"], key=lambda c: c["repetitions"])
        c_hi = canaries[-1]
        exp = canary_exposure(
            model, tokenizer, c_hi["phrase"], args.max_length, device,
            n_candidates=args.canary_candidates, seed=args.seed,
        )
        recon = attack_a4_reconstruction(
            model, tokenizer, [c for c in corpus["train"] if "CONFIDENTIAL PHI MARKER" in c][:5],
            corpus["public"], args.max_length, device,
        )
        util = eval_utility_proxy(model, tokenizer, corpus["holdout"], args.max_length, device)
        row = {
            "r": r,
            "params_m": meta["lora_params"] / 1e6,
            "macro_f1_proxy": util["utility_score"],
            "mean_nll": util["mean_nll"],
            "mia_auc": a1["auc"],
            "exposure": exp["exposure"],
            "edit_distance": recon["edit_distance"],
        }
        rows.append(row)
        logger.info("rank row %s", row)
        del model
        torch.cuda.empty_cache()
    out = os.path.join(args.output_dir, "rank_sweep.json")
    with open(out, "w") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def run_recon(args, corpus):
    settings = [
        ("ep1_n10k", 1, min(args.n_train, 10000)),
        ("ep10_n10k", 10, min(args.n_train, 10000)),
        ("ep10_n100", 10, 100),
    ]
    # For smoke / default n_train, remap labels to actual sizes
    rows = []
    for tag, epochs, n in settings:
        n = min(n, len(corpus["train"]))
        texts = corpus["train"][:n]
        # Ensure canaries present in small set
        if n <= 100:
            canary_texts = [t for t in corpus["train"] if "CONFIDENTIAL PHI MARKER" in t]
            texts = (canary_texts + texts)[:n]
        path = train_one(args, texts, f"recon_{tag}", lora_rank=args.base_rank, epochs=epochs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, cfg, meta = load_adapted_model(args.model_name, path, device)
        targets = [t for t in texts if "CONFIDENTIAL PHI MARKER" in t][:10]
        recon = attack_a4_reconstruction(
            model, tokenizer, targets, corpus["public"], args.max_length, device
        )
        exposures = {}
        for c in corpus["canaries"]:
            if c["repetitions"] in (1, 16):
                exposures[f"x{c['repetitions']}"] = canary_exposure(
                    model, tokenizer, c["phrase"], args.max_length, device,
                    n_candidates=args.canary_candidates, seed=args.seed,
                )
        row = {
            "setting": tag,
            "epochs": epochs,
            "n": n,
            **recon,
            "exposures": exposures,
        }
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    out = os.path.join(args.output_dir, "recon.json")
    with open(out, "w") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def run_defenses(args, corpus):
    # Baseline adapter
    base_path = train_one(args, corpus["train"], "def_none", lora_rank=args.base_rank, epochs=args.epochs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    members = corpus["train"][: args.n_mia]
    nonmembers = corpus["holdout"][: args.n_mia]

    def evaluate_adapter(path, label, mutate=None):
        model, tokenizer, cfg, meta = load_adapted_model(args.model_name, path, device)
        if mutate:
            mutate(model)
        a1 = attack_a1_base_calibrated(
            model, tokenizer, members, nonmembers, args.max_length, device
        )
        targets = [t for t in corpus["train"] if "CONFIDENTIAL PHI MARKER" in t][:8]
        recon = attack_a4_reconstruction(
            model, tokenizer, targets, corpus["public"], args.max_length, device
        )
        util = eval_utility_proxy(model, tokenizer, corpus["holdout"], args.max_length, device)
        del model
        torch.cuda.empty_cache()
        return {
            "defense": label,
            "macro_f1_proxy": util["utility_score"],
            "mean_nll": util["mean_nll"],
            "mia_auc": a1["auc"],
            "phi_recall": recon["phi_recall"],
            "edit_distance": recon["edit_distance"],
        }

    rows = []
    rows.append(evaluate_adapter(base_path, "none"))

    # D2 truncation
    rows.append(evaluate_adapter(base_path, "d2_trunc_r2", mutate=lambda m: defense_d2_truncate(m, 2)))

    # D3 frozen-A
    path_d3 = train_one(
        args, corpus["train"], "def_frozen_a", lora_rank=args.base_rank, epochs=args.epochs, freeze_a=True
    )
    rows.append(evaluate_adapter(path_d3, "d3_frozen_a"))

    # D4 merge of k shard adapters
    k = 4 if args.stage == "smoke" else 8
    shard_states = []
    shard_size = max(1, len(corpus["train"]) // k)
    for i in range(k):
        shard = corpus["train"][i * shard_size : (i + 1) * shard_size]
        if not shard:
            shard = corpus["train"][: max(1, shard_size)]
        p = train_one(args, shard, f"def_shard_{i}", lora_rank=args.base_rank, epochs=max(1, args.epochs))
        blob = torch.load(p, map_location="cpu")
        shard_states.append(blob["lora_state"])
    merged = defense_d4_merge_state_dicts(shard_states, method="ties")
    merge_dir = adapter_dir(args.output_dir, "def_merge_ties")
    os.makedirs(merge_dir, exist_ok=True)
    base_blob = torch.load(base_path, map_location="cpu")
    base_blob["lora_state"] = merged
    merge_path = os.path.join(merge_dir, "adapter.pt")
    torch.save(base_blob, merge_path)
    rows.append(evaluate_adapter(merge_path, "d4_merge_ties"))

    # D5 AdaptGuard
    teacher, tokenizer, cfg, meta = load_adapted_model(args.model_name, base_path, device)
    n_steps = 50 if args.stage == "smoke" else 200
    student_state = defense_d5_adaptguard(
        teacher, tokenizer, corpus["public"], cfg, device, n_steps=n_steps
    )
    del teacher
    torch.cuda.empty_cache()
    d5_dir = adapter_dir(args.output_dir, "def_adaptguard")
    os.makedirs(d5_dir, exist_ok=True)
    blob = torch.load(base_path, map_location="cpu")
    blob["lora_state"] = student_state
    d5_path = os.path.join(d5_dir, "adapter.pt")
    torch.save(blob, d5_path)
    rows.append(evaluate_adapter(d5_path, "d5_adaptguard"))

    # D1 DP-LoRA
    for eps in (8.0, 1.0):
        p = train_one(
            args,
            corpus["train"],
            f"def_dp_eps{int(eps)}",
            lora_rank=args.base_rank,
            epochs=args.epochs,
            dp_epsilon=eps,
        )
        rows.append(evaluate_adapter(p, f"d1_dp_eps{int(eps)}"))

    out = os.path.join(args.output_dir, "defenses.json")
    with open(out, "w") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    if args.stage == "smoke":
        args.n_train = 64
        args.n_holdout = 64
        args.n_public = 32
        args.n_mia = 32
        args.epochs = 1
        args.ranks = "1,4,16"
        args.max_length = 256
        args.canary_candidates = 64
        args.model_name = args.model_name  # keep, but user may override to 1B

    corpus = ensure_corpus(args)
    if args.stage == "data":
        return

    if args.stage in ("train", "all", "smoke"):
        train_one(args, corpus["train"], f"main_r{args.base_rank}", lora_rank=args.base_rank, epochs=args.epochs)

    main_adapter = os.path.join(adapter_dir(args.output_dir, f"main_r{args.base_rank}"), "adapter.pt")

    if args.stage in ("mia", "all", "smoke"):
        if not os.path.exists(main_adapter):
            train_one(args, corpus["train"], f"main_r{args.base_rank}", lora_rank=args.base_rank, epochs=args.epochs)
        run_mia(args, corpus, main_adapter, tag="mia_main")

    if args.stage in ("rank", "all", "smoke"):
        run_rank_sweep(args, corpus)

    if args.stage in ("recon", "all", "smoke"):
        run_recon(args, corpus)

    if args.stage in ("defense", "all", "smoke"):
        run_defenses(args, corpus)

    # Summary pointer
    summary = {"output_dir": args.output_dir, "model": args.model_name, "stage": args.stage}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Done stage=%s", args.stage)


if __name__ == "__main__":
    main()
