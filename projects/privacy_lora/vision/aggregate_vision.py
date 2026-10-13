"""Aggregate vision MedMNIST results and print a LaTeX-ready summary."""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT_S = os.environ.get("PRIVACY_LORA_VISION", "privacy_lora_v2/vision_superposition")
ROOT_M = os.environ.get("PRIVACY_LORA_VISION_MIA", "privacy_lora_v2/vision_mia")
DS = ("pathmnist", "dermamnist", "pneumoniamnist")
T1 = [
    "sgd_r16_t1", "sgd_r16_t10", "sgd_r16_t50", "sgd_r16_t200", "sgd_r16_t500",
    "sgd_r1_t50", "sgd_r4_t50", "frozenA_r16_t50",
]
T2 = ["known_A0_sparse", "A_T_proxy", "indep_init", "no_sparsity", "perturb_1pct"]


def main():
    by = defaultdict(list)
    for p in sorted(glob.glob(f"{ROOT_S}/*_s*.json")):
        d = json.load(open(p))
        by[d["dataset"]].append(d)
    print(f"superposition runs: {sum(len(v) for v in by.values())}/9")
    if sum(len(v) for v in by.values()) >= 1:
        print("\n=== Table-1 style (means) ===")
        for lab in T1:
            parts = []
            for ds in DS:
                recs = [r["table1"][lab] for r in by[ds] if lab in r.get("table1", {})]
                if not recs:
                    parts.append(f"{ds}: —")
                    continue
                r = np.mean([x["R_ff"] for x in recs])
                s = np.mean([x["supp"] for x in recs])
                e = np.mean([x["E_read"] for x in recs])
                parts.append(f"{ds}: R={r:.3f} S={s:.0f} E={e:.3f}")
            print(lab, " | ".join(parts))
        print("\n=== Table-2 style ===")
        for lab in T2:
            parts = []
            for ds in DS:
                recs = [r["table2"][lab] for r in by[ds] if lab in r.get("table2", {})]
                if not recs:
                    continue
                ec = np.mean([x["E_c"] for x in recs])
                s = np.mean([x["supp"] for x in recs])
                rf = np.mean([x["R_fit"] for x in recs])
                parts.append(f"{ds}: Ec={ec:.3f} S={s:.0f} Rf={rf:.3f}")
            if parts:
                print(lab, " | ".join(parts))

    print("\n=== Vision MIA ===")
    for p in sorted(glob.glob(f"{ROOT_M}/*_s*.json")):
        d = json.load(open(p))
        print(f"{d['dataset']} s{d['seed']}: loss_AUC={d['auc_loss']:.3f} A1_AUC={d['auc_a1']:.3f} ({d['clinical']})")


if __name__ == "__main__":
    main()
