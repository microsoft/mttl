"""Fill Table 1–2 \\resTBD cells in NAACL_privacy_lora.tex from superposition_v3 JSONs."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np

ROOT = os.environ.get("PRIVACY_LORA_SUPERPOS", "privacy_lora_v2/superposition_v3")
TEX = "NAACL_privacy_lora.tex"
SEEDS = (0, 1, 2)
MODEL = "qwen14b"

T1_ORDER = [
    "sgd_r16_t1",
    "sgd_r16_t10",
    "sgd_r16_t50",
    "sgd_r16_t200",
    "sgd_r16_t500",
    "sgd_r1_t50",
    "sgd_r4_t50",
    "frozenA_r16_t50",
]
T2_ORDER = ["known_A0_sparse", "A_T_proxy", "indep_init", "no_lora_proj", "no_sparsity", "perturb_1pct"]
DS = ("pmc", "synthetic", "enron")


def load(model: str) -> Dict[str, List[Dict]]:
    by = {ds: [] for ds in DS}
    for ds in DS:
        for s in SEEDS:
            p = os.path.join(ROOT, f"{model}_{ds}_s{s}.json")
            if os.path.exists(p):
                by[ds].append(json.load(open(p)))
    return by


def fmt(m: float, key: str) -> str:
    if key == "supp":
        return f"{m:.0f}"
    if abs(m) >= 10:
        return f"{m:.2f}"
    if abs(m) >= 0.01 or m == 0.0:
        return f"{m:.3f}"
    return f"{m:.2e}"


def mean_cell(runs: List[Dict], label: str, key: str) -> Optional[str]:
    vals = []
    for r in runs:
        rec = r.get("table1", {}).get(label)
        if rec and key in rec:
            vals.append(float(rec[key]))
    if not vals:
        return None
    return fmt(float(np.mean(vals)), key)


def mean_t2(runs: List[Dict], label: str, key: str) -> Optional[str]:
    vals = []
    for r in runs:
        rec = r.get("table2", {}).get(label)
        if rec and key in rec:
            vals.append(float(rec[key]))
    if not vals:
        return None
    return fmt(float(np.mean(vals)), key)


def triple(a, b, c):
    if a is None or b is None or c is None:
        return None
    return f"{a} & {b} & {c}"


def row9(by, label):
    cells = []
    for ds in DS:
        cells.append(triple(
            mean_cell(by[ds], label, "R_ff"),
            mean_cell(by[ds], label, "supp"),
            mean_cell(by[ds], label, "E_read"),
        ))
    if any(c is None for c in cells):
        return None
    return " & ".join(cells)


def row_t2(by, label):
    cells = []
    for ds in DS:
        cells.append(triple(
            mean_t2(by[ds], label, "E_c"),
            mean_t2(by[ds], label, "supp"),
            mean_t2(by[ds], label, "R_fit"),
        ))
    if any(c is None for c in cells):
        return None
    return " & ".join(cells)


def replace_tbd_block(text: str, start_pat: str, n_rows: int, new_rows: List[str]) -> str:
    m = re.search(start_pat, text)
    if not m:
        raise SystemExit(f"pattern not found: {start_pat}")
    pos = m.end()
    chunk = text[pos:]
    pattern = re.compile(r"^.*\\resTBD.*$\n", re.M)
    hits = list(pattern.finditer(chunk))
    if len(hits) < n_rows:
        raise SystemExit(f"only {len(hits)} TBD rows after {start_pat}")
    start = pos + hits[0].start()
    end = pos + hits[n_rows - 1].end()
    return text[:start] + "".join(new_rows) + text[end:]


def main():
    by = load(MODEL)
    n = sum(len(v) for v in by.values())
    print(f"loaded {MODEL} runs={n}/9 from {ROOT}")
    if n < 9:
        print("incomplete; not rewriting tex")
        return

    labels1 = [
        "SGD & 16 & 1",
        "SGD & 16 & 10",
        "SGD & 16 & 50",
        "SGD & 16 & 200",
        "SGD & 16 & 500",
        "SGD & 1  & 50",
        "SGD & 4  & 50",
        "Frozen-$A$ SGD & 16 & 50",
    ]
    rows = []
    for lab, key in zip(labels1, T1_ORDER):
        r = row9(by, key)
        if r is None:
            print("missing table1", key)
            return
        rows.append(f"{lab}  & {r} \\\\\n")

    t2_labels = [
        r"Known $A_0$ + sparse fit",
        r"Final $A_T$ as a proxy",
        r"Independent initialization",
        r"No LoRA projection",
        r"No sparsity constraint",
        r"Perturbed observation (1\%)",
    ]
    rows_t2 = []
    for lab, key in zip(t2_labels, T2_ORDER):
        r = row_t2(by, key)
        if r is None:
            print("missing table2", key)
            return
        rows_t2.append(f"{lab} & {r} \\\\\n")

    tex = open(TEX).read()
    tex = replace_tbd_block(tex, r"\\textbf\{Qwen2\.5-14B-Instruct\}", 8, rows)
    tex = replace_tbd_block(tex, r"\\label\{tab:recovery_ablation\}", 6, rows_t2)
    tex = tex.replace(
        "All response cells are new experiments and remain \\resTBD.",
        "Entries are means over three seeds. Support is the percent of seeds with exact 4-of-20 recovery.",
    )
    tex = tex.replace(
        "All response cells are \\resTBD.",
        "Entries are means over the same three Qwen2.5-14B seeds as the $r{=}16$, 50-step row above.",
    )
    open(TEX, "w").write(tex)
    print("updated", TEX)
    print("remaining \\resTBD:", tex.count("\\resTBD"))


if __name__ == "__main__":
    main()
