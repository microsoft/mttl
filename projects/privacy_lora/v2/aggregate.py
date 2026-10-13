"""Aggregate v2 results into LaTeX table snippets and figures.

    /usr/bin/python3 projects/privacy_lora/v2/aggregate.py \
        --results privacy_lora_v2/results --out privacy_lora_v2/paper

Only numpy + matplotlib are required (runs on the login node).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEEDS = (0, 1, 2)
Z95 = 1.96
FPR_GRID = np.logspace(-4, 0, 33)

ATTACK_LABELS = [
    ("loss", "Loss threshold"),
    ("zlib", "Zlib"),
    ("mink", "Min-$K$\\%"),
    ("minkpp", "Min-$K$\\%++"),
    ("ref", "A1: base-calibrated (ref.)"),
    ("ref_mink", "A1 + Min-$K$\\%"),
    ("a2_A", "A2: subspace ($A$)"),
    ("a2_A-A0", "A2: subspace ($A{-}A_0$)"),
    ("a2_B", "A2: gradient side ($B$)"),
    ("fusion", "Fusion (A1 $+$ A2)"),
    ("fusion_mink", "Fusion (A1$_{\\text{Min-}K}$ $+$ A2)"),
]
BLIND_LABELS = [("blind_len", "Blind: length"), ("blind_zlib", "Blind: zlib ratio"), ("blind_base_loss", "Blind: base-model loss")]


# --------------------------------------------------------------------------- #
class Results:
    def __init__(self, root: str):
        self.R: Dict[str, Dict] = {}
        for p in sorted(glob.glob(os.path.join(root, "*.json"))):
            name = os.path.basename(p)[:-5]
            try:
                self.R[name] = json.load(open(p))
            except Exception as e:  # noqa: BLE001
                print("could not load", p, e)
        self.root = root

    def has(self, name: str) -> bool:
        return name in self.R

    def evals(self, name: str) -> Dict:
        return self.R[name]["evals"]

    def tags(self, name: str) -> List[str]:
        return list(self.evals(name).keys())

    def ev(self, name: str, tag: str) -> Optional[Dict]:
        """Evaluation record; fields missing from the main arm (A2, exposure, recon at the early-stop
        epoch) are filled from the matching early-stop re-evaluation arm (``*_es_s{seed}``)."""
        e = self.evals(name).get(tag)
        sib = name.replace("_r16_s", "_r16_es_s") if "_es_" not in name and "_r16_s" in name else None
        if sib and sib in self.R and tag in self.evals(sib):
            es = self.evals(sib)[tag]
            if e is None:
                return es
            merged = dict(es)
            merged.update(e)
            if "mia" in es:
                merged["mia"] = {**es["mia"], **e.get("mia", {})}
            return merged
        return e

    def early_tag(self, name: str) -> str:
        tr = self.R[name].get("train", {})
        e = tr.get("early_stop_epoch")
        tags = [t for t in self.tags(name) if t.startswith("ep") and "mia" in self.evals(name)[t]]
        if e is None:
            cands = [(self.evals(name)[t].get("train", {}).get("val_nll"), int(t[2:])) for t in tags]
            cands = [c for c in cands if c[0] is not None]
            if cands:
                e = min(cands)[1]
        if e is not None and f"ep{e}" in tags:
            return f"ep{e}"
        # closest evaluated epoch <= e
        eps = sorted(int(t[2:]) for t in tags)
        if e is not None and eps:
            below = [x for x in eps if x <= e]
            return f"ep{max(below) if below else eps[0]}"
        return tags[-1] if tags else "ep3"

    def last_tag(self, name: str) -> str:
        tags = [t for t in self.tags(name) if t.startswith("ep") and "mia" in self.evals(name)[t]]
        return f"ep{max(int(t[2:]) for t in tags)}" if tags else "final"

    def mia(self, name: str, tag: str, attack: str, key: str = "auc") -> Optional[Tuple[float, float]]:
        e = self.ev(name, tag)
        if not e or "mia" not in e or attack not in e["mia"]:
            return None
        m = e["mia"][attack]
        return m[key], m.get(key + "_boot_sd", 0.0)

    def util(self, name: str, tag: str, key: str = "val_nll") -> Optional[float]:
        e = self.ev(name, tag)
        return e["utility"].get(key) if e and "utility" in e else None

    def exposure(self, name: str, tag: str, reps: int = 16, key: str = "exposure_extrapolated") -> Optional[float]:
        e = self.ev(name, tag)
        if not e or "exposure" not in e:
            return None
        for x in e["exposure"]:
            if x["repetitions"] == reps:
                return x[key]
        return None

    def decoy_exposure(self, name: str, tag: str, key: str = "exposure_extrapolated") -> Optional[float]:
        e = self.ev(name, tag)
        if not e or "exposure_decoys" not in e:
            return None
        return float(np.mean([x[key] for x in e["exposure_decoys"]]))

    def recon(self, name: str, tag: str, cfg: str = "greedy") -> Optional[Dict]:
        e = self.ev(name, tag)
        if not e or "recon" not in e or cfg not in e["recon"]:
            return None
        return e["recon"][cfg]

    def gpu_hours(self) -> float:
        return sum(r.get("train", {}).get("gpu_hours", 0.0) for r in self.R.values())


# --------------------------------------------------------------------------- #
def combine(vals: Sequence[Tuple[float, float]]) -> Tuple[float, float, int]:
    """Mean over seeds and 95% half-width combining bootstrap SD and seed variance."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    pts = np.array([v[0] for v in vals])
    sds = np.array([v[1] for v in vals])
    k = len(pts)
    se = math.sqrt(np.mean(sds**2) / k + (pts.var(ddof=1) / k if k > 1 else 0.0))
    return float(pts.mean()), Z95 * se, k


def f3(mean: float, hw: Optional[float] = None, pct: bool = False, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if pct:
        s = f"{100 * mean:.1f}"
        return s if hw is None or math.isnan(hw) else f"{s}{{\\scriptsize$\\pm${100 * hw:.1f}}}"
    s = f"{mean:.{digits}f}"
    return s if hw is None or math.isnan(hw) else f"{s}{{\\scriptsize$\\pm${hw:.{digits}f}}}"


def seed_names(pattern: str, R: Results, seeds=SEEDS) -> List[str]:
    return [pattern.format(s=s) for s in seeds if R.has(pattern.format(s=s))]


def cell_mia(R: Results, names: List[str], tagf, attack: str, key: str = "auc") -> str:
    vals = [R.mia(n, tagf(n), attack, key) for n in names]
    m, hw, k = combine(vals)
    return f3(m, hw, pct=key != "auc")


def mean_of(vals: List[Optional[float]]) -> Tuple[float, float]:
    v = [x for x in vals if x is not None]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), (Z95 * float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else float("nan"))


def write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# Table 1: MIA main
# --------------------------------------------------------------------------- #
def table_mia(R: Results, out: str, prefix: str = "L8_syn", tag_label="") -> Dict:
    lm = seed_names(prefix + "_lm_r16_s{s}", R)
    inst = seed_names(prefix + "_inst_r16_s{s}", R)
    cols = [("LM early-stop", lm, R.early_tag), ("LM 10 ep", lm, R.last_tag), ("Inst.\\ early-stop", inst, R.early_tag), ("Inst.\\ 10 ep", inst, R.last_tag)]
    cols = [c for c in cols if c[1]]
    lines = ["\\begin{tabular}{l" + "cc" * len(cols) + "}", "\\toprule",
             "& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{c[0]}}}" for c in cols) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
             "Attack & " + " & ".join("AUC & TPR@1\\%" for _ in cols) + " \\\\", "\\midrule"]
    for key, label in ATTACK_LABELS:
        cells = []
        for _, names, tagf in cols:
            cells += [cell_mia(R, names, tagf, key, "auc"), cell_mia(R, names, tagf, key, "tpr_at_1pct_fpr")]
        if all(c == "--" for c in cells):
            continue
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    for key, label in BLIND_LABELS:
        cells = []
        for _, names, tagf in cols:
            cells += [cell_mia(R, names, tagf, key, "auc"), cell_mia(R, names, tagf, key, "tpr_at_1pct_fpr")]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    utl = []
    for _, names, tagf in cols:
        nll = mean_of([R.util(n, tagf(n)) for n in names])
        ep = mean_of([float(tagf(n)[2:]) for n in names])
        utl += [f"\\multicolumn{{2}}{{c}}{{{f3(nll[0], digits=3)} / ep {ep[0]:.1f}}}"]
    lines.append("Held-out NLL / epoch & " + " & ".join(utl) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, f"tab_mia{tag_label}.tex"), "\n".join(lines))
    info = {"lm_early_epochs": [R.early_tag(n) for n in lm], "inst_early_epochs": [R.early_tag(n) for n in inst]}
    return info


# --------------------------------------------------------------------------- #
# Table 2: rank sweep with TOST
# --------------------------------------------------------------------------- #
def tost(R: Results, names_a: List[str], names_b: List[str], tag_a, tag_b, attack="ref", margin=0.05) -> Tuple[str, float, float]:
    """Paired-by-seed difference AUC(a) - AUC(b); 90% CI inside (-margin, margin) => equivalent."""
    diffs, sds = [], []
    for a, b in zip(names_a, names_b):
        va, vb = R.mia(a, tag_a(a), attack), R.mia(b, tag_b(b), attack)
        if va is None or vb is None:
            continue
        diffs.append(va[0] - vb[0])
        sds.append(math.sqrt(va[1] ** 2 + vb[1] ** 2))
    if not diffs:
        return "--", float("nan"), float("nan")
    k = len(diffs)
    se = math.sqrt(np.mean(np.array(sds) ** 2) / k + (np.var(diffs, ddof=1) / k if k > 1 else 0.0))
    lo, hi = np.mean(diffs) - 1.645 * se, np.mean(diffs) + 1.645 * se
    verdict = "$\\equiv$" if (lo > -margin and hi < margin) else ("$\\neq$" if (lo > 0 or hi < 0) else "n.s.")
    return verdict, lo, hi


def e10_names(R: Results, P: str, r: int) -> List[str]:
    return seed_names(P + ("_r16_s{s}" if r == 16 else f"_r{r}_e10_s{{s}}"), R)


def table_rank(R: Results, out: str, prefix="L8_pmc_lm", fname="tab_rank.tex"):
    ranks = [1, 2, 4, 8, 16, 64]
    ref3 = seed_names(prefix + "_r16_s{s}", R)
    ref10 = e10_names(R, prefix, 16)
    ep3 = lambda n: "ep3"  # noqa: E731
    lines = ["\\begin{tabular}{rr|cccc|ccccc|c}", "\\toprule",
             "& & \\multicolumn{4}{c|}{Matched epochs (3)} & \\multicolumn{5}{c|}{Matched utility (early-stop)} & \\\\",
             "$r$ & Params (M) & NLL & AUC (A1) & TPR@1\\% & TOST vs.\\ $r{=}16$ & ep & NLL & AUC (A1) & TPR@1\\% & TOST vs.\\ $r{=}16$ & Expo.$_{\\times16}$ \\\\", "\\midrule"]
    rows = {}
    for r in ranks:
        names3 = seed_names(prefix + f"_r{r}_s{{s}}", R)
        names10 = e10_names(R, prefix, r)
        if not names3 and not names10:
            continue
        any_name = (names3 or names10)[0]
        params = R.R[any_name]["method"]["n_trainable"] / 1e6
        nll3 = mean_of([R.util(n, "ep3") for n in names3])
        tv = tost(R, names3, ref3, ep3, ep3) if names3 else ("--", float("nan"), float("nan"))
        t3 = "--" if (r == 16 or tv[0] == "--") else "{} [{:+.3f}, {:+.3f}]".format(*tv)
        c3 = f"{f3(nll3[0], nll3[1])} & {cell_mia(R, names3, ep3, 'ref')} & {cell_mia(R, names3, ep3, 'ref', 'tpr_at_1pct_fpr')} & {t3}" if names3 else "-- & -- & -- & --"
        if names10:
            ep = mean_of([float(R.early_tag(n)[2:]) for n in names10])
            nll10 = mean_of([R.util(n, R.early_tag(n)) for n in names10])
            tv = tost(R, names10, ref10, R.early_tag, R.early_tag)
            t10 = "--" if (r == 16 or tv[0] == "--") else "{} [{:+.3f}, {:+.3f}]".format(*tv)
            c10 = f"{ep[0]:.1f} & {f3(nll10[0], nll10[1])} & {cell_mia(R, names10, R.early_tag, 'ref')} & {cell_mia(R, names10, R.early_tag, 'ref', 'tpr_at_1pct_fpr')} & {t10}"
        else:
            c10 = "-- & -- & -- & -- & --"
        ex = mean_of([R.exposure(n, "ep3") for n in names3]) if names3 else (float("nan"), float("nan"))
        rows[r] = dict(auc3=combine([R.mia(n, "ep3", "ref") for n in names3]), nll3=nll3)
        lines.append(f"{r} & {params:.2f} & {c3} & {c10} & {f3(ex[0], ex[1], digits=1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, fname), "\n".join(lines))
    return rows


# --------------------------------------------------------------------------- #
# Table: method comparison (E5)
# --------------------------------------------------------------------------- #
def table_method(R: Results, out: str, P="L8_pmc_lm", fname="tab_method.tex"):
    methods = [("Full fine-tuning", P + "_full_s{s}", P + "_full_s{s}"), ("LoRA $r{=}1$", P + "_r1_s{s}", P + "_r1_e10_s{s}"),
               ("LoRA $r{=}16$", P + "_r16_s{s}", P + "_r16_s{s}"), ("LoRA $r{=}64$", P + "_r64_s{s}", P + "_r64_e10_s{s}"),
               ("IA$^3$", P + "_ia3_s{s}", P + "_ia3_e10_s{s}"), ("Head-only", P + "_head_s{s}", P + "_head_e10_s{s}")]
    ep3 = lambda n: "ep3"  # noqa: E731
    lines = ["\\begin{tabular}{lr|cccc|cccc}", "\\toprule",
             "& & \\multicolumn{4}{c|}{Matched epochs (3)} & \\multicolumn{4}{c}{Matched utility (early-stop)} \\\\",
             "Method & Params (M) & NLL & AUC (A1) & TPR@1\\% & AUC (Loss) & ep & NLL & AUC (A1) & TPR@1\\% \\\\", "\\midrule"]
    for label, pat3, pat10 in methods:
        names3 = seed_names(pat3, R)
        names10 = seed_names(pat10, R) or names3
        if not names3 and not names10:
            continue
        params = R.R[(names3 or names10)[0]]["method"]["n_trainable"] / 1e6
        nll3 = mean_of([R.util(n, "ep3") for n in names3])
        c3 = f"{f3(nll3[0], nll3[1])} & {cell_mia(R, names3, ep3, 'ref')} & {cell_mia(R, names3, ep3, 'ref', 'tpr_at_1pct_fpr')} & {cell_mia(R, names3, ep3, 'loss')}" if names3 else "-- & -- & -- & --"
        ep = mean_of([float(R.early_tag(n)[2:]) for n in names10])
        nll10 = mean_of([R.util(n, R.early_tag(n)) for n in names10])
        c10 = f"{ep[0]:.1f} & {f3(nll10[0], nll10[1])} & {cell_mia(R, names10, R.early_tag, 'ref')} & {cell_mia(R, names10, R.early_tag, 'ref', 'tpr_at_1pct_fpr')}"
        lines.append(f"{label} & {params:,.1f} & {c3} & {c10} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, fname), "\n".join(lines))


# --------------------------------------------------------------------------- #
# Table: defenses (unified config)
# --------------------------------------------------------------------------- #
def defense_rows(R: Results, prefix: str = "L8_syn_lm", seeds=SEEDS) -> List[Tuple[str, List[str], object, str]]:
    def one(label, pat, tag, guar="--"):
        names = seed_names(pat, R, seeds)
        return (label, names, (lambda n, t=tag: t), guar)

    rows = [one("None (LoRA $r{=}16$)", prefix + "_r16_s{s}", "ep3")]
    for r in (1, 2, 4, 8):
        rows.append(one(f"D2: rank truncation $r'{{=}}{r}$", prefix + "_d2_s{s}", f"d2_r{r}"))
    rows.append(one("D3: frozen-$A$", prefix + "_d3_s{s}", "ep3"))
    rows.append(one("D4: shard merge, TIES ($k{=}8$)", prefix + "_d4_s{s}", "d4_ties"))
    rows.append(one("D4: shard merge, uniform ($k{=}8$)", prefix + "_d4_s{s}", "d4_uniform"))
    rows.append(one("D5: \\textsc{AdaptGuard}", prefix + "_d5_s{s}", "final"))
    rows.append(one("Public-data-only adapter (no private data)", prefix + "_d5_pubonly_s{s}", "final"))
    rows.append((r"D1: $\varepsilon{=}\infty$ control (same $B$, lr, early-stop)", seed_names(prefix + "_dpinf_s{s}", R, seeds), R.early_tag, "--"))
    rows.append(one("D1: DP-LoRA $\\varepsilon{=}8$", prefix + "_dp8_s{s}", "ep20", "$(8,\\delta)$"))
    rows.append(one("D1: DP-LoRA $\\varepsilon{=}8$, $B{=}512$", prefix + "_dp8_b512_s{s}", "ep20", "$(8,\\delta)$"))
    rows.append(one("D1: DP-LoRA $\\varepsilon{=}3$", prefix + "_dp3_s{s}", "ep20", "$(3,\\delta)$"))
    rows.append(one("D1: DP-LoRA $\\varepsilon{=}1$", prefix + "_dp1_s{s}", "ep20", "$(1,\\delta)$"))
    return [r for r in rows if r[1]]


def table_defense(R: Results, out: str, prefix="L8_syn_lm", fname="tab_defense.tex"):
    rows = defense_rows(R, prefix)
    base_name = seed_names(prefix + "_r16_s{s}", R)
    lines = ["\\begin{tabular}{lcccccccc}", "\\toprule",
             "Defense & NLL & AUC (A1) & TPR@1\\% (A1) & AUC (Loss) & AUC (A2) & Expo.$_{\\times16}$ & PHI rec. & Guar. \\\\", "\\midrule"]
    if base_name:
        b = [R.R[n]["base"] for n in base_name]
        nll = mean_of([x["utility"]["val_nll"] for x in b])
        ex = mean_of([next((e["exposure_extrapolated"] for e in x.get("exposure", []) if e["repetitions"] == 16), None) for x in b])
        lines.append(f"Base model (no adapter) & {f3(nll[0], nll[1])} & 0.500 & 1.0 & 0.500 & -- & {f3(ex[0], ex[1], digits=2)} & 0.0 & -- \\\\")
        lines.append("\\midrule")
    scatter = []
    for label, names, tagf, guar in rows:
        nll = mean_of([R.util(n, tagf(n)) for n in names])
        ex = mean_of([R.exposure(n, tagf(n)) for n in names])
        rec = mean_of([(R.recon(n, tagf(n)) or {}).get("canary_phi_recall") for n in names])
        auc = combine([R.mia(n, tagf(n), "ref") for n in names])
        scatter.append((label, nll[0], auc[0], auc[1]))
        lines.append(f"{label} & {f3(nll[0], nll[1])} & {cell_mia(R, names, tagf, 'ref')} & {cell_mia(R, names, tagf, 'ref', 'tpr_at_1pct_fpr')} & "
                     f"{cell_mia(R, names, tagf, 'loss')} & {cell_mia(R, names, tagf, 'a2_A')} & {f3(ex[0], ex[1], digits=2)} & {f3(rec[0], None, pct=True)} & {guar} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, fname), "\n".join(lines))
    return scatter


def table_d5_ablation(R: Results, out: str):
    P = "L8_pmc_lm"
    rows = [("\\textsc{AdaptGuard} default ($|\\D_{\\text{pub}}|{=}10^3$, top-50, $\\sigma{=}0.5$)", P + "_d5_s0"),
            ("$\\D_{\\text{pub}}$-only fresh adapter (no teacher)", P + "_d5_pubonly_s0"),
            ("Distillation, no top-$k$, no noise", P + "_d5_nonoise_s0"),
            ("Logit noise $\\sigma{=}0.1$", P + "_d5_noise0.1_s0"),
            ("Logit noise $\\sigma{=}2.0$", P + "_d5_noise2_s0"),
            ("Top-$k{=}10$", P + "_d5_topk10_s0"),
            ("$|\\D_{\\text{pub}}|{=}100$", P + "_d5_pub100_s0"),
            ("$|\\D_{\\text{pub}}|{=}300$", P + "_d5_pub300_s0"),
            ("$|\\D_{\\text{pub}}|{=}3000$", P + "_d5_pub3000_s0"),
            ("$\\D_{\\text{pub}}$ from MTSamples (shifted)", P + "_d5_shift_s0")]
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule", "Variant & NLL & AUC (A1) & TPR@1\\% & AUC (A2) & Expo.$_{\\times16}$ & PHI rec.\\ (adaptive) \\\\", "\\midrule"]
    t = R.ev(P + "_d5_s0", "teacher") if R.has(P + "_d5_s0") else None
    if t:
        rec = (t.get("recon") or {}).get("greedy", {}).get("canary_phi_recall")
        lines.append(f"Teacher (undefended, released) & {f3(t['utility']['val_nll'])} & {f3(t['mia']['ref']['auc'], Z95 * t['mia']['ref']['auc_boot_sd'])} & "
                     f"{f3(t['mia']['ref']['tpr_at_1pct_fpr'], pct=True)} & {f3(t['mia'].get('a2_A', {}).get('auc'))} & "
                     f"{f3(R.exposure(P + '_d5_s0', 'teacher'), digits=2)} & {f3(rec, pct=True) if rec is not None else '--'} \\\\")
        lines.append("\\midrule")
    for label, n in rows:
        if not R.has(n):
            continue
        e = R.ev(n, "final")
        if not e:
            continue
        rec = (e.get("recon") or {}).get("greedy", {}).get("canary_phi_recall")
        lines.append(f"{label} & {f3(e['utility']['val_nll'])} & {f3(e['mia']['ref']['auc'], Z95 * e['mia']['ref']['auc_boot_sd'])} & "
                     f"{f3(e['mia']['ref']['tpr_at_1pct_fpr'], pct=True)} & {f3(e['mia'].get('a2_A', {}).get('auc'))} & "
                     f"{f3(R.exposure(n, 'final'), digits=2)} & {f3(rec, pct=True) if rec is not None else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_d5_ablation.tex"), "\n".join(lines))


# --------------------------------------------------------------------------- #
# Backbones / real data / SGD / variants / size / memorisation / A5 / recon / exposure
# --------------------------------------------------------------------------- #
def table_backbone(R: Results, out: str):
    bbs = [("Llama-3.2-1B", "llama1b"), ("Llama-3.2-3B", "llama3b"), ("Mistral-7B-Instruct-v0.3", "mistral7b"), ("BioMistral-7B", "biomistral7b"),
           ("Qwen2.5-7B-Instruct", "qwen7b"), ("Qwen2.5-14B-Instruct", "qwen14b"),
           ("Qwen2.5-32B-Instruct", "qwen32b"), ("Llama-3.1-8B-Instruct", "L8")]
    lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
             "Backbone & Early-stop ep & NLL & AUC (A1) & TPR@1\\% & AUC (A1) 10 ep & AUC $r{=}1$ & AUC $r{=}64$ \\\\", "\\midrule"]
    for label, key in bbs:
        n = f"{key}_pmc_lm_r16_s0"
        if not R.has(n):
            continue
        et = R.early_tag(n)
        r1, r64 = f"{key}_pmc_lm_r1_s0", f"{key}_pmc_lm_r64_s0"
        c = lambda name, tag, k="auc": (f3(*(lambda v: (v[0], Z95 * v[1]))(R.mia(name, tag, "ref", k)), pct=k != "auc") if R.has(name) and R.mia(name, tag, "ref", k) else "--")  # noqa: E731
        lines.append(f"{label} & {et[2:]} & {f3(R.util(n, et))} & {c(n, et)} & {c(n, et, 'tpr_at_1pct_fpr')} & {c(n, R.last_tag(n))} & {c(r1, 'ep3')} & {c(r64, 'ep3')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_backbone.tex"), "\n".join(lines))


def table_real(R: Results, out: str):
    lines = ["\\begin{tabular}{llccccccc}", "\\toprule",
             "Corpus & Setting & ep & NLL & AUC (A1) & TPR@1\\% & AUC (Loss) & AUC (A2) & Blind AUC \\\\", "\\midrule"]

    def row(corpus, label, names, tagf):
        if not names:
            return
        ep = mean_of([float(tagf(n)[2:]) if tagf(n).startswith("ep") else float("nan") for n in names])
        nll = mean_of([R.util(n, tagf(n)) for n in names])
        blind = cell_mia(R, names, tagf, "blind_base_loss")
        lines.append(f"{corpus} & {label} & {ep[0]:.0f} & {f3(nll[0], nll[1])} & {cell_mia(R, names, tagf, 'ref')} & {cell_mia(R, names, tagf, 'ref', 'tpr_at_1pct_fpr')} & "
                     f"{cell_mia(R, names, tagf, 'loss')} & {cell_mia(R, names, tagf, 'a2_A')} & {blind} \\\\")

    pmc = seed_names("L8_pmc_lm_r16_s{s}", R)
    row("PMC-Patients", "LM, early-stop", pmc, R.early_tag)
    row("PMC-Patients", "LM, 10 ep", pmc, R.last_tag)
    row("PMC-Patients", "LM, 3 ep", pmc, lambda n: "ep3")
    inst = seed_names("L8_pmc_inst_r16_s{s}", R)
    row("PMC-Patients", "Inst., early-stop", inst, R.early_tag)
    row("PMC-Patients", "Inst., 10 ep", inst, R.last_tag)
    for r in (1, 64):
        row("PMC-Patients", f"LM, $r{{=}}{r}$, 3 ep", seed_names(f"L8_pmc_lm_r{r}_s{{s}}", R), lambda n: "ep3")
    en = seed_names("L8_enron_lm_r16_s{s}", R)
    row("Enron e-mail", "LM, early-stop", en, R.early_tag)
    row("Enron e-mail", "LM, 10 ep", en, R.last_tag)
    syn = seed_names("L8_syn_lm_r16_s{s}", R)
    row("Synthetic", "LM, early-stop", syn, R.early_tag)
    row("Synthetic", "LM, 10 ep", syn, R.last_tag)
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_real.tex"), "\n".join(lines))


def table_sgd(R: Results, out: str, P="L8_pmc_lm", fname="tab_sgd.tex"):
    rows = [("AdamW", P + "_r16_s{s}", "ep3"), ("SGD ($\\beta{=}0.9$)", P + "_r16_sgd_s{s}", "ep3"),
            ("AdamW, frozen-$A$", P + "_d3_s{s}", "ep3"), ("SGD, frozen-$A$", P + "_r16_sgd_d3_s{s}", "ep3")]
    lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
             "Optimizer & NLL & Loss & A1 & A2 ($A$) & A2 ($A{-}A_0$) & A2 ($B$-side) & Fusion \\\\", "\\midrule"]
    for label, pat, tag in rows:
        names = seed_names(pat, R)
        if not names:
            continue
        nll = mean_of([R.util(n, tag) for n in names])
        t = lambda n: tag  # noqa: E731
        lines.append(f"{label} & {f3(nll[0], nll[1])} & {cell_mia(R, names, t, 'loss')} & {cell_mia(R, names, t, 'ref')} & {cell_mia(R, names, t, 'a2_A')} & "
                     f"{cell_mia(R, names, t, 'a2_A-A0')} & {cell_mia(R, names, t, 'a2_B')} & {cell_mia(R, names, t, 'fusion')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, fname), "\n".join(lines))


def table_variants(R: Results, out: str):
    rows = [("Attention $q,k,v,o$ (default)", "L8_pmc_lm_r16_s0"), ("MLP gate/up/down", "L8_pmc_lm_r16_mlp_s0"), ("All linear", "L8_pmc_lm_r16_all_s0"),
            ("rsLoRA scaling $\\alpha/\\sqrt{r}$", "L8_pmc_lm_r16_rslora_s0")]
    lines = ["\\begin{tabular}{lrcccc}", "\\toprule", "Target / variant & Params (M) & NLL & AUC (A1) & TPR@1\\% & AUC (A2) \\\\", "\\midrule"]
    for label, n in rows:
        if not R.has(n):
            continue
        v = R.mia(n, "ep3", "ref")
        t = R.mia(n, "ep3", "ref", "tpr_at_1pct_fpr")
        a2 = R.mia(n, "ep3", "a2_A")
        if v is None:
            continue
        lines.append(f"{label} & {R.R[n]['method']['n_trainable'] / 1e6:.1f} & {f3(R.util(n, 'ep3'))} & {f3(v[0], Z95 * v[1])} & {f3(t[0], Z95 * t[1], pct=True)} & {f3(a2[0], Z95 * a2[1]) if a2 else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_variants.tex"), "\n".join(lines))


def table_size(R: Results, out: str):
    lines = ["\\begin{tabular}{rlccccc}", "\\toprule", "$|\\D|$ & Checkpoint & NLL & AUC (Loss) & AUC (A1) & TPR@1\\% (A1) & AUC (A2) \\\\", "\\midrule"]
    for size, pat in [(100, "L8_pmc_lm_r16_n100_s{s}"), (1000, "L8_pmc_lm_r16_s{s}"), (10000, "L8_pmc_lm_r16_n10000_s{s}")]:
        names = seed_names(pat, R)
        if not names:
            continue
        for label, tagf in [("early-stop", R.early_tag), ("3 ep", lambda n: "ep3"), ("last", R.last_tag)]:
            if size == 10000 and label == "last":
                continue
            nll = mean_of([R.util(n, tagf(n)) for n in names])
            ep = " (ep " + "/".join(tagf(n)[2:] for n in names) + ")" if label == "early-stop" else ""
            lines.append(f"{size} & {label}{ep} & {f3(nll[0], nll[1])} & {cell_mia(R, names, tagf, 'loss')} & {cell_mia(R, names, tagf, 'ref')} & "
                         f"{cell_mia(R, names, tagf, 'ref', 'tpr_at_1pct_fpr')} & {cell_mia(R, names, tagf, 'a2_A')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_size.tex"), "\n".join(lines))


def table_memorize(R: Results, out: str):
    rows = [(f"{ds}: $r{{=}}1$, one layer (L16 $o$-proj)", f"L8_mem{sfx}_r1_layer16_oproj") for ds, sfx in (("PMC", "_pmc"), ("Synthetic", ""))]
    rows += [(f"{ds}: $r{{=}}1$, all attention", f"L8_mem{sfx}_r1_attn") for ds, sfx in (("PMC", "_pmc"), ("Synthetic", ""))]
    rows += [(f"{ds}: $r{{=}}16$, all attention", f"L8_mem{sfx}_r16_attn") for ds, sfx in (("PMC", "_pmc"), ("Synthetic", ""))]
    lines = ["\\begin{tabular}{lrcccc}", "\\toprule", "Adapter & Params & Steps & Token match & Longest prefix & Held-out NLL \\\\", "\\midrule"]
    for label, n in rows:
        if not R.has(n):
            continue
        for tag in sorted(R.tags(n), key=lambda t: int(t[2:]) if t[2:].isdigit() else 0):
            e = R.ev(n, tag)
            if not e or "dup_probe" not in e:
                continue
            d = e["dup_probe"][0]
            ctrl = np.mean([c["token_match_frac"] for c in e["dup_probe_control"]])
            lines.append(f"{label} & {R.R[n]['method']['n_trainable']:,} & {tag[2:]} & {d['token_match_frac']:.2f} (ctrl {ctrl:.2f}) & {d['longest_prefix_match']}/{d['n_ref']} & {f3(e['utility']['val_nll'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_memorize.tex"), "\n".join(lines))


def table_a5(R: Results, out: str):
    rows = [("Cohort A (75\\% female, skewed dx)", "L8_synA_lm_r16_a5_s0"), ("Cohort 0 (50\\% female, uniform dx)", "L8_syn_lm_r16_s0"), ("Cohort B (25\\% female, uniform dx)", "L8_synB_lm_r16_a5_s0")]
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule", "Training cohort & True female & Est.\\ (adapter) & Est.\\ (base) & TV(dx) adapter & TV(dx) base & $n$ parsed \\\\", "\\midrule"]
    for label, n in rows:
        if not R.has(n):
            continue
        tag = "ep10" if "ep10" in R.tags(n) else "ep3"
        e = R.ev(n, tag)
        if not e or "a5" not in e:
            continue
        a = e["a5"]
        est, base = a["estimate"], a.get("base", {})
        lines.append(f"{label} & {a['truth']['female_frac']:.2f} & {est['female_frac']:.2f} & {base.get('female_frac', float('nan')):.2f} & "
                     f"{f3(a.get('dx_tv'))} & {f3(a.get('base_dx_tv'))} & {est['n_parsed_sex']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_a5.tex"), "\n".join(lines))


def table_recon(R: Results, out: str):
    arms = [("None (LoRA-16, 10 ep)", "L8_syn_lm_r16_s0", "ep10"), ("None (LoRA-16, 3 ep, $r{=}1$)", "L8_syn_lm_r1_s0", "ep3"), ("None ($r{=}64$, 3 ep)", "L8_syn_lm_r64_s0", "ep3"),
            ("D2 $r'{=}2$", "L8_syn_lm_d2_s0", "d2_r2"), ("D3 frozen-$A$", "L8_syn_lm_d3_s0", "ep3"), ("D4 TIES", "L8_syn_lm_d4_s0", "d4_ties"),
            ("D5 \\textsc{AdaptGuard}", "L8_syn_lm_d5_s0", "final"), ("Full FT (3 ep)", "L8_syn_lm_full_s0", "ep3"),
            ("PMC-Patients (10 ep)", "L8_pmc_lm_r16_s0", "ep10"), ("Enron (10 ep)", "L8_enron_lm_r16_s0", "ep10")]
    lines = ["\\begin{tabular}{llcccc}", "\\toprule", "Target adapter & Search & Canary PHI recall & Decoy recall & Member IDs & Non-member IDs \\\\", "\\midrule"]
    for label, n, tag in arms:
        if not R.has(n) or not R.ev(n, tag) or "recon" not in R.ev(n, tag):
            continue
        for cfg, clabel in [("greedy", "template greedy"), ("gcg_lm", "GCG, LM term"), ("gcg_sub", "GCG, subspace term"), ("gcg_full", "GCG, both (Eq.~5)")]:
            m = R.recon(n, tag, cfg)
            if not m:
                continue
            ids_m = m["member_mrn_hits"] + m["member_email_hits"]
            ids_n = m["nonmember_mrn_hits"] + m["nonmember_email_hits"]
            lines.append(f"{label} & {clabel} & {100 * m['canary_phi_recall']:.0f}\\% & {100 * m['decoy_phi_recall']:.0f}\\% & {ids_m} & {ids_n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_recon.tex"), "\n".join(lines))


def table_exposure(R: Results, out: str):
    P = "L8_pmc_lm"
    arms = [("LoRA-16, early-stop (2 ep)", P + "_r16_s{s}", "ep2"), ("LoRA-16, 3 ep", P + "_r16_s{s}", "ep3"), ("LoRA-16, 10 ep", P + "_r16_s{s}", "ep10"),
            ("LoRA-1, 3 ep", P + "_r1_s{s}", "ep3"), ("LoRA-64, 3 ep", P + "_r64_s{s}", "ep3"), ("Full FT, 3 ep", P + "_full_s{s}", "ep3"),
            ("D3 frozen-$A$, 3 ep", P + "_d3_s{s}", "ep3"), ("D4 merge (uniform)", P + "_d4_s{s}", "d4_uniform"), ("D5 \\textsc{AdaptGuard}", P + "_d5_s{s}", "final"),
            ("DP $\\varepsilon{=}8$", P + "_dp8_s{s}", "ep20"),
            ("Synthetic, LoRA-16, 10 ep", "L8_syn_lm_r16_s{s}", "ep10"), ("Synthetic, full FT, 3 ep", "L8_syn_lm_full_s{s}", "ep3")]
    lines = ["\\begin{tabular}{lcccccc|c}", "\\toprule", "Adapter & $\\times1$ & $\\times2$ & $\\times4$ & $\\times8$ & $\\times16$ & Decoys ($\\times0$) & Dup.\\ note $\\times16$ match \\\\", "\\midrule"]
    for label, pat, tag in arms:
        names = seed_names(pat, R)
        if not names:
            continue
        cells = []
        for reps in (1, 2, 4, 8, 16):
            v = mean_of([R.exposure(n, tag, reps) for n in names])
            cells.append(f3(v[0], v[1], digits=1))
        dec = mean_of([R.decoy_exposure(n, tag) for n in names])
        dup = mean_of([next((d["token_match_frac"] for d in (R.ev(n, tag) or {}).get("dup_probe", []) if d["reps"] == 16), None) for n in names])
        lines.append(f"{label} & " + " & ".join(cells) + f" & {f3(dec[0], dec[1], digits=1)} & {f3(dup[0], dup[1], digits=2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tab_exposure.tex"), "\n".join(lines))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def figures(R: Results, out: str, scatter):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5})
    # AUC vs epoch
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    for ax, (title, pat) in zip(axes, [("PMC-Patients, causal LM", "L8_pmc_lm_r16_s{s}"), ("PMC-Patients, note completion", "L8_pmc_inst_r16_s{s}")]):
        names = seed_names(pat, R)
        if not names:
            continue
        ax2 = ax.twinx()
        for attack, lab, c in [("loss", "Loss", "C0"), ("minkpp", "Min-K%++", "C2"), ("ref", "A1 (ref.)", "C3"), ("a2_A", "A2", "C4"), ("fusion", "Fusion", "C1")]:
            xs, ys, es = [], [], []
            for ep in range(1, 11):
                vals = [R.mia(n, f"ep{ep}", attack) for n in names]
                vals = [v for v in vals if v]
                if vals:
                    xs.append(ep); ys.append(np.mean([v[0] for v in vals])); es.append(np.std([v[0] for v in vals]))
            if xs:
                ax.errorbar(xs, ys, yerr=es, label=lab, color=c, marker="o", ms=2.5, lw=1, capsize=1.5)
        nll = []
        for ep in range(1, 11):
            v = [R.util(n, f"ep{ep}") for n in names if R.ev(n, f"ep{ep}")]
            nll.append(np.mean(v) if v else np.nan)
        ax2.plot(range(1, 11), nll, color="gray", ls="--", lw=1, label="held-out NLL")
        ax2.set_ylabel("held-out NLL", color="gray")
        es_eps = [int(R.early_tag(n)[2:]) for n in names]
        for e in set(es_eps):
            ax.axvline(e, color="k", ls=":", lw=0.8)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylim(0.4, 1.02)
        ax.legend(loc="lower right", ncol=2)
    axes[0].set_ylabel("AUC")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_auc_vs_epoch.pdf")); plt.close(fig)

    # log-log ROC (seed 0, LM early-stop and 10 ep)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4))
    for ax, (title, name, tag) in zip(axes, [("LM, early-stop", "L8_pmc_lm_r16_s0", None), ("LM, 10 epochs", "L8_pmc_lm_r16_s0", "ep10")]):
        if not R.has(name):
            continue
        tag = tag or R.early_tag(name)
        e = R.ev(name, tag)
        if not e or "mia" not in e:
            continue
        for attack, lab, c in [("loss", "Loss", "C0"), ("minkpp", "Min-K%++", "C2"), ("ref", "A1", "C3"), ("a2_A", "A2", "C4"), ("fusion", "Fusion", "C1"), ("blind_base_loss", "Blind (base loss)", "gray")]:
            if attack in e["mia"]:
                ax.plot(FPR_GRID, np.maximum(e["mia"][attack]["roc_tpr"], 1e-4), label=lab, lw=1, color=c)
        ax.plot([1e-4, 1], [1e-4, 1], "k:", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
        ax.set_xlabel("FPR"); ax.set_title(f"{title} ({tag})")
    axes[0].set_ylabel("TPR"); axes[0].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_roc_loglog.pdf")); plt.close(fig)

    # privacy-utility scatter: defenses vs. the undefended training trajectory
    if scatter:
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        main = seed_names("L8_pmc_lm_r16_s{s}", R)
        xs, ys = [], []
        for ep in range(1, 7):
            v = [R.mia(n, f"ep{ep}", "ref") for n in main]
            u = [R.util(n, f"ep{ep}") for n in main]
            v = [a[0] for a in v if a]; u = [a for a in u if a is not None]
            if v and u:
                xs.append(np.mean(u)); ys.append(np.mean(v))
                ax.annotate(f"ep{ep}", (xs[-1], ys[-1]), fontsize=5, color="gray", xytext=(-3, 3), textcoords="offset points", ha="right")
        if xs:
            ax.plot(xs, ys, "-", color="gray", lw=1, marker=".", ms=4, label="no defense, $r{=}16$, epochs 1--6")
        shorts = {"None": "none (3 ep)", "D2: rank truncation $r'{=}1$": "D2 $r'{=}1$", "D2: rank truncation $r'{=}2$": "D2 $r'{=}2$",
                  "D2: rank truncation $r'{=}4$": "D2 $r'{=}4$", "D2: rank truncation $r'{=}8$": "D2 $r'{=}8$", "D3: frozen-$A$": "D3",
                  "D4: shard merge, TIES ($k{=}8$)": "D4 TIES", "D4: shard merge, uniform ($k{=}8$)": "D4 unif.", "D5: \\textsc{AdaptGuard}": "D5", "Public-data-only adapter (no private data)": "public-only",
                  "D1: DP-LoRA $\\varepsilon{=}8$": "DP $\\varepsilon{=}8$", "D1: DP-LoRA $\\varepsilon{=}8$, $B{=}512$": "DP $\\varepsilon{=}8$, $B{=}512$",
                  "D1: DP-LoRA $\\varepsilon{=}3$": "DP $\\varepsilon{=}3$", "D1: DP-LoRA $\\varepsilon{=}1$": "DP $\\varepsilon{=}1$"}
        fam_color = {"D1": "C3", "D2": "C0", "D3": "C2", "D4": "C4", "D5": "C1", "Pu": "C5"}
        offsets = {"None": (3, 3), "D2: rank truncation $r'{=}8$": (3, -7), "D2: rank truncation $r'{=}4$": (-3, -7)}
        for label, nll, auc, hw in scatter:
            if nll is None or math.isnan(nll) or math.isnan(auc):
                continue
            key = next((k for k in shorts if label.startswith(k)), None)
            if key is None:
                continue
            fam = key[:2] if key[:2] in fam_color else "D0"
            ax.errorbar(nll, auc, yerr=0 if math.isnan(hw) else hw, fmt="o", ms=3.5, capsize=1.5, color=fam_color.get(fam, "k"))
            dx, dy = offsets.get(key, (3, -1))
            ax.annotate(shorts[key], (nll, auc), fontsize=5.5, xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left")
        ax.set_xlabel("held-out NLL (lower is better)"); ax.set_ylabel("A1 AUC (lower is safer)")
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.legend(loc="upper right", fontsize=5.5)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_privacy_utility.pdf")); plt.close(fig)

    # rank curve (matched epochs vs matched utility)
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    for lab, getn, tagf, ls in [("A1, 3 epochs", lambda r: seed_names(f"L8_pmc_lm_r{r}_s{{s}}", R), lambda n: "ep3", "-"),
                                ("A1, early-stop", lambda r: e10_names(R, "L8_pmc_lm", r), R.early_tag, "--"),
                                ("Loss, 3 epochs", lambda r: seed_names(f"L8_pmc_lm_r{r}_s{{s}}", R), lambda n: "ep3", ":")]:
        attack = "loss" if lab.startswith("Loss") else "ref"
        xs, ys, es = [], [], []
        for r in (1, 2, 4, 8, 16, 64):
            names = getn(r)
            m, hw, k = combine([R.mia(n, tagf(n), attack) for n in names])
            if k:
                xs.append(r); ys.append(m); es.append(hw)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, lw=1, capsize=2, label=lab, ls=ls)
    ax.set_xscale("log", base=2); ax.set_xlabel("LoRA rank $r$"); ax.set_ylabel("AUC"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_rank.pdf")); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="privacy_lora_v2/results")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--figs", default="figures")
    args = ap.parse_args()
    R = Results(args.results)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)
    print(f"{len(R.R)} arms loaded; {R.gpu_hours():.1f} GPU-hours")
    meta = {"n_arms": len(R.R), "gpu_hours": R.gpu_hours(), "arms": sorted(R.R)}
    meta["mia"] = table_mia(R, args.out, prefix="L8_pmc")
    meta["mia_syn"] = table_mia(R, args.out, prefix="L8_syn", tag_label="_syn")
    table_rank(R, args.out)
    table_rank(R, args.out, prefix="L8_syn_lm", fname="tab_rank_syn.tex")
    table_method(R, args.out)
    table_method(R, args.out, P="L8_syn_lm", fname="tab_method_syn.tex")
    scatter = table_defense(R, args.out, prefix="L8_pmc_lm")
    table_defense(R, args.out, prefix="L8_syn_lm", fname="tab_defense_syn.tex")
    table_d5_ablation(R, args.out)
    table_backbone(R, args.out)
    table_real(R, args.out)
    table_sgd(R, args.out)
    table_sgd(R, args.out, P="L8_syn_lm", fname="tab_sgd_syn.tex")
    table_variants(R, args.out)
    table_size(R, args.out)
    table_memorize(R, args.out)
    table_a5(R, args.out)
    table_recon(R, args.out)
    table_exposure(R, args.out)
    try:
        figures(R, args.figs, scatter)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("figures failed:", e)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)


if __name__ == "__main__":
    main()
