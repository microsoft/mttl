"""Generate the experiment grid (one JSONL file per Slurm job) for the naacl.md checklist.

Primary corpus: PMC-Patients (real case reports + surrogate PHI header).
Controlled corpus: synthetic discharge summaries (canaries, cohorts, memorisation demos).

    python projects/privacy_lora/v2/make_specs.py --out privacy_lora_v2/specs
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

L8 = "meta-llama/Llama-3.1-8B-Instruct"
BACKBONES = {
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "biomistral7b": "BioMistral/BioMistral-7B",
    "llama1b": "meta-llama/Llama-3.2-1B",
    "llama3b": "meta-llama/Llama-3.2-3B",
}
ALL10 = list(range(1, 11))
FULL10 = [1, 3, 5, 10]
DS = {"pmc": "pmc", "syn": "synthetic"}


def arm(name: str, ds: str = "pmc", **kw) -> Dict:
    d = {"name": name, "backbone": L8, "dataset": DS[ds], "fmt": "lm", "method": "lora", "rank": 16,
         "epochs": 3, "eval_epochs": [1, 2, 3], "full_eval_epochs": [3], "seed": 0, "data_seed": 0}
    d.update(kw)
    return d


MEM_EPOCHS = [50, 100, 200, 300]


def mem_arms(ds: str):
    """E17: single-note memorisation demo (one held-out duplicate note, no other data)."""
    common = dict(memorize=True, batch=1, micro_batch=1, epochs=300, eval_epochs=MEM_EPOCHS, full_eval_epochs=MEM_EPOCHS,
                  exposure=False, n_boot=0, lora_dropout=0.0)
    sfx = "" if ds == "syn" else f"_{ds}"
    return [arm(f"L8_mem{sfx}_r1_layer16_oproj", ds, rank=1, modules_regex=".*layers\\.16\\.self_attn", targets="o_proj", lr=2e-3, **common),
            arm(f"L8_mem{sfx}_r1_attn", ds, rank=1, lr=1e-3, **common),
            arm(f"L8_mem{sfx}_r16_attn", ds, rank=16, lr=1e-3, **common)]


def main_arm(name: str, seed: int, ds: str = "pmc", **kw) -> Dict:
    base = dict(seed=seed, data_seed=seed, epochs=10, eval_epochs=ALL10, full_eval_epochs=FULL10,
                recon="gcg" if seed == 0 else "greedy", b_side=(seed == 0), a5=(seed == 0 and ds == "syn"))
    base.update(kw)
    return arm(name, ds, **base)


def dp_arm(name: str, ds: str, eps: float, seed: int = 0, **kw) -> Dict:
    base = dict(defense="dp", dp_epsilon=eps, dp_batch=256, lr=1e-3, epochs=20, eval_epochs=[5, 10, 20], full_eval_epochs=[20],
                seed=seed, data_seed=seed)
    base.update(kw)
    return arm(name, ds, **base)


def build() -> Dict[str, List[Dict]]:
    J: Dict[str, List[Dict]] = {}
    for ds in ("pmc", "syn"):
        P = f"L8_{ds}"
        # main LM arms, 3 seeds, per-epoch audit
        for s in range(3):
            J[f"{ds}_main_lm_s{s}"] = [main_arm(f"{P}_lm_r16_s{s}", s, ds)]
        # rank sweep (r16 = main arm ep3)
        for s in range(3):
            J[f"{ds}_rank_s{s}"] = [arm(f"{P}_lm_r{r}_s{s}", ds, rank=r, seed=s, data_seed=s, recon="greedy" if s == 0 else "none")
                                    for r in (1, 2, 4, 8, 64)]
        # defenses at the unified config
        for s in range(3):
            J[f"{ds}_def_s{s}"] = [
                arm(f"{P}_lm_d2_s{s}", ds, defense="d2", d2_ranks=[1, 2, 4, 8], seed=s, data_seed=s, recon="greedy" if s == 0 else "none"),
                arm(f"{P}_lm_d3_s{s}", ds, freeze_a=True, seed=s, data_seed=s, b_side=(s == 0), recon="greedy" if s == 0 else "none"),
                arm(f"{P}_lm_d4_s{s}", ds, defense="d4", d4_shards=8, seed=s, data_seed=s, recon="greedy" if s == 0 else "none"),
                arm(f"{P}_lm_d5_s{s}", ds, defense="d5", d5_n_public=1000, seed=s, data_seed=s, recon="gcg" if s == 0 else "none"),
            ]
            J[f"{ds}_dp8_s{s}"] = [dp_arm(f"{P}_lm_dp8_s{s}", ds, 8.0, s)]
        J[f"{ds}_dp_sweep"] = [dp_arm(f"{P}_lm_dp1_s0", ds, 1.0), dp_arm(f"{P}_lm_dp3_s0", ds, 3.0)]
        J[f"{ds}_dp_extra"] = [
            dp_arm(f"{P}_lm_dp8_b512_s0", ds, 8.0, dp_batch=512, eval_epochs=[10, 20]),
            arm(f"{P}_lm_dpinf_s0", ds, batch=256, lr=1e-3, epochs=20, eval_epochs=[5, 10, 20], full_eval_epochs=[20],
                notes="eps=inf control: same batch/lr/epochs as DP arms, no clipping/noise"),
        ]
        # SGD vs AdamW (subspace analysis)
        J[f"{ds}_sgd"] = [arm(f"{P}_lm_r16_sgd_s{s}", ds, optimizer="sgd", lr=1e-3, seed=s, data_seed=s, b_side=(s == 0)) for s in range(3)] + [
            arm(f"{P}_lm_r16_sgd_d3_s0", ds, optimizer="sgd", lr=1e-3, freeze_a=True, b_side=True)]

    # ---- PMC-only parts of the grid (primary corpus) ----
    P = "L8_pmc"
    for s in range(3):
        J[f"pmc_main_inst_s{s}"] = [main_arm(f"{P}_inst_r16_s{s}", s, "pmc", fmt="inst")]
        J[f"pmc_method_s{s}"] = [
            arm(f"{P}_lm_full_s{s}", "pmc", method="full", lr=1e-5, seed=s, data_seed=s, recon="greedy" if s == 0 else "none"),
            arm(f"{P}_lm_ia3_s{s}", "pmc", method="ia3", lr=3e-3, seed=s, data_seed=s),
            arm(f"{P}_lm_head_s{s}", "pmc", method="head", lr=1e-4, seed=s, data_seed=s),
        ]
    J["pmc_size"] = [arm(f"{P}_lm_r16_n100_s{s}", "pmc", n_train=100, n_members_eval=100, seed=s, data_seed=s, epochs=10,
                         eval_epochs=ALL10, full_eval_epochs=[1, 3, 10]) for s in range(3)]
    J["pmc_size10k"] = [arm(f"{P}_lm_r16_n10000_s0", "pmc", n_train=10000, epochs=3, eval_epochs=[1, 2, 3], full_eval_epochs=[1, 3])]

    def d5(n, **kw):
        base = {"defense": "d5", "d5_n_public": 1000, "recon": "greedy"}
        base.update(kw)
        return arm(n, "pmc", **base)

    J["pmc_d5_abl_a"] = [d5(f"{P}_lm_d5_pubonly_s0", d5_pub_only=True), d5(f"{P}_lm_d5_nonoise_s0", d5_noise=0.0, d5_top_k=0),
                         d5(f"{P}_lm_d5_noise0.1_s0", d5_noise=0.1), d5(f"{P}_lm_d5_noise2_s0", d5_noise=2.0)]
    J["pmc_d5_abl_b"] = [d5(f"{P}_lm_d5_topk10_s0", d5_top_k=10), d5(f"{P}_lm_d5_pub100_s0", d5_n_public=100),
                         d5(f"{P}_lm_d5_pub300_s0", d5_n_public=300), d5(f"{P}_lm_d5_pub3000_s0", d5_n_public=3000),
                         d5(f"{P}_lm_d5_shift_s0", d5_public="shift")]
    for key, bb in BACKBONES.items():
        mb = 2 if "32b" in key else 4
        J[f"pmc_bb_{key}"] = [
            main_arm(f"{key}_pmc_lm_r16_s0", 0, "pmc", backbone=bb, b_side=False, recon="greedy", micro_batch=mb),
            arm(f"{key}_pmc_lm_r1_s0", "pmc", backbone=bb, rank=1, micro_batch=mb),
            arm(f"{key}_pmc_lm_r64_s0", "pmc", backbone=bb, rank=64, micro_batch=mb),
        ]
    # Extra seeds on the 14B Qwen (same protocol as Llama-8B main arm).
    for s in (1, 2):
        J[f"pmc_bb_qwen14b_s{s}"] = [
            main_arm(f"qwen14b_pmc_lm_r16_s{s}", s, "pmc", backbone=BACKBONES["qwen14b"], b_side=False, recon="greedy"),
        ]
    J["pmc_variants"] = [arm(f"{P}_lm_r16_mlp_s0", "pmc", targets="mlp"), arm(f"{P}_lm_r16_all_s0", "pmc", targets="all"),
                         arm(f"{P}_lm_r16_rslora_s0", "pmc", variant="rslora")]
    J["enron"] = [main_arm("L8_enron_lm_r16_s0", 0, "pmc", dataset="enron", b_side=False, recon="greedy", exposure=False, dup_probe=False)]

    # ---- synthetic-only parts (controlled canaries / cohorts / memorisation) ----
    S = "L8_syn"
    J["syn_method"] = [arm(f"{S}_lm_full_s0", "syn", method="full", lr=1e-5, recon="greedy"), arm(f"{S}_lm_ia3_s0", "syn", method="ia3", lr=3e-3),
                       arm(f"{S}_lm_head_s0", "syn", method="head", lr=1e-4)]
    J["syn_d5_abl"] = [arm(f"{S}_lm_d5_pubonly_s0", "syn", defense="d5", d5_pub_only=True, recon="greedy"),
                       arm(f"{S}_lm_d5_nonoise_s0", "syn", defense="d5", d5_noise=0.0, d5_top_k=0, recon="greedy"),
                       arm(f"{S}_lm_d5_shift_s0", "syn", defense="d5", d5_public="shift", recon="greedy")]
    J["syn_bb"] = [main_arm(f"{key}_syn_lm_r16_s0", 0, "syn", backbone=bb, b_side=False, recon="greedy", a5=False)
                   for key, bb in BACKBONES.items() if key in ("qwen7b", "mistral7b", "llama3b")]
    J["syn_cohort_mem"] = [
        arm("L8_synA_lm_r16_a5_s0", "syn", female_frac=0.75, dx_skew=3.0, a5=True, exposure=False, dup_probe=False),
        arm("L8_synB_lm_r16_a5_s0", "syn", female_frac=0.25, dx_skew=0.0, a5=True, exposure=False, dup_probe=False),
    ] + mem_arms("syn") + mem_arms("pmc")
    return J


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="privacy_lora_v2/specs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.out):
        if f.endswith(".jsonl") and f != "smoke.jsonl":
            os.remove(os.path.join(args.out, f))
    jobs = build()
    names = [a["name"] for arms in jobs.values() for a in arms]
    assert len(names) == len(set(names)), "duplicate arm names"
    for job, arms in jobs.items():
        with open(os.path.join(args.out, job + ".jsonl"), "w") as f:
            for a in arms:
                f.write(json.dumps(a) + "\n")
    print(f"{len(jobs)} jobs, {len(names)} arms -> {args.out}")
