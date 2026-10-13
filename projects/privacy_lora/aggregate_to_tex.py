"""Aggregate privacy_lora JSON results and patch iclr2027_privacy_lora.tex tables."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}\\%"


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def capacity_numbers() -> Dict[str, str]:
    # Llama-3.1-8B attention q/k/v/o, r=16, bf16
    params = 32 * 16 * ((4096 + 4096) + (4096 + 1024) + (4096 + 1024) + (4096 + 4096))
    params_m = params / 1e6
    mb = params * 2 / 1e6
    ratio = (mb * 1e6) / 1536.0
    return {
        "params_m": f"{params_m:.1f}",
        "mb": f"{mb:.1f}",
        "ratio": f"{ratio:.0f}",
    }


def theoretical_params_m(r: int, model: str = "llama31_8b") -> float:
    if model == "llama32_1b":
        # 16 layers, GQA: q 2048->2048, k/v 2048->512, o 2048->2048
        per = r * ((2048 + 2048) + (2048 + 512) + (2048 + 512) + (2048 + 2048))
        return 16 * per / 1e6
    # Llama-3.1-8B attention q/k/v/o
    per = r * ((4096 + 4096) + (4096 + 1024) + (4096 + 1024) + (4096 + 4096))
    return 32 * per / 1e6


def build_replacements(root: str) -> Dict[str, str]:
    rep: Dict[str, str] = {}
    cap = capacity_numbers()
    rep["CAP_PARAMS_M"] = cap["params_m"]
    rep["CAP_MB"] = cap["mb"]
    rep["CAP_RATIO"] = cap["ratio"]

    # Always fill theoretical param counts for the rank table.
    for r in (1, 2, 4, 8, 16, 64):
        rep[f"R{r}_PARAMS"] = fmt(theoretical_params_m(r), 2)

    mia = load_json(os.path.join(root, "mia_main.json"))
    if mia:
        for key, alias in [
            ("loss_threshold", "LOSS"),
            ("mink", "MINK"),
            ("a1", "A1"),
            ("a2", "A2"),
            ("a1_a2_fusion", "FUSE"),
            ("merged_control", "MERGED"),
        ]:
            block = mia.get(key) or {}
            rep[f"{alias}_AUC"] = fmt(block.get("auc", float("nan")))
            rep[f"{alias}_TPR1"] = pct(block.get("tpr_at_1pct_fpr", 0.0))
            rep[f"{alias}_TPR01"] = pct(block.get("tpr_at_0_1pct_fpr", 0.0))

    rank = load_json(os.path.join(root, "rank_sweep.json"))
    if rank:
        for row in rank:
            r = row["r"]
            # Keep theoretical PARAMS; fill empirical columns from the sweep.
            rep[f"R{r}_F1"] = fmt(row["macro_f1_proxy"], 3)
            rep[f"R{r}_AUC"] = fmt(row["mia_auc"], 3)
            rep[f"R{r}_EXP"] = fmt(row["exposure"], 2)
            rep[f"R{r}_EDIT"] = fmt(row["edit_distance"], 3)

    recon = load_json(os.path.join(root, "recon.json"))
    if recon:
        by = {row["setting"]: row for row in recon}
        for setting, prefix in [
            ("ep1_n10k", "RECON1"),
            ("ep10_n10k", "RECON10"),
            ("ep10_n100", "RECON100"),
        ]:
            row = by.get(setting)
            if not row:
                continue
            rep[f"{prefix}_EDIT"] = fmt(row["edit_distance"], 3)
            rep[f"{prefix}_EXACT"] = pct(row["exact_span_rate"])
            rep[f"{prefix}_PHI"] = pct(row["phi_recall"])
            expos = row.get("exposures") or {}
            if "x1" in expos:
                rep["CANARY1_EXP"] = fmt(expos["x1"]["exposure"], 2)
            if "x16" in expos:
                rep["CANARY16_EXP"] = fmt(expos["x16"]["exposure"], 2)
            # also put per-setting exposure from x16 if present
            if "x16" in expos:
                rep[f"{prefix}_EXP"] = fmt(expos["x16"]["exposure"], 2)
            elif "x1" in expos:
                rep[f"{prefix}_EXP"] = fmt(expos["x1"]["exposure"], 2)

    defs = load_json(os.path.join(root, "defenses.json"))
    if defs:
        key_map = {
            "none": "DEF_NONE",
            "d2_trunc_r2": "DEF_D2",
            "d3_frozen_a": "DEF_D3",
            "d4_merge_ties": "DEF_D4",
            "d5_adaptguard": "DEF_D5",
            "d1_dp_eps8": "DEF_DP8",
            "d1_dp_eps1": "DEF_DP1",
        }
        for row in defs:
            prefix = key_map.get(row["defense"])
            if not prefix:
                continue
            rep[f"{prefix}_F1"] = fmt(row["macro_f1_proxy"], 3)
            rep[f"{prefix}_AUC"] = fmt(row["mia_auc"], 3)
            rep[f"{prefix}_PHI"] = pct(row["phi_recall"])

        # Abstract deltas
        none = next((r for r in defs if r["defense"] == "none"), None)
        d5 = next((r for r in defs if r["defense"] == "d5_adaptguard"), None)
        dp8 = next((r for r in defs if r["defense"] == "d1_dp_eps8"), None)
        if none and d5:
            rep["ABS_D5_RECON"] = pct(d5["phi_recall"])
            rep["ABS_D5_F1DROP"] = fmt(100.0 * (none["macro_f1_proxy"] - d5["macro_f1_proxy"]), 1)
        if none and dp8:
            rep["ABS_DP8_F1DROP"] = fmt(100.0 * (none["macro_f1_proxy"] - dp8["macro_f1_proxy"]), 1)

    if mia and "a1" in mia:
        rep["ABS_TPR01"] = pct(mia["a1"].get("tpr_at_0_1pct_fpr", 0.0))
    if rank and len(rank) >= 2:
        byr = {row["r"]: row for row in rank}
        if 64 in byr and 2 in byr:
            rep["ABS_RANK_DELTA"] = fmt(100.0 * abs(byr[64]["mia_auc"] - byr[2]["mia_auc"]), 1)

    return rep


TABLE_MIA = r"""
\begin{tabular}{lccc}
\toprule
Attack & AUC & TPR@1\%FPR & TPR@0.1\%FPR \\
\midrule
Random & 0.500 & 1.0\% & 0.1\% \\
Loss threshold \citep{yeom2018privacy} & __LOSS_AUC__ & __LOSS_TPR1__ & __LOSS_TPR01__ \\
Min-$K$\% \citep{shi2024detecting} & __MINK_AUC__ & __MINK_TPR1__ & __MINK_TPR01__ \\
A1 base-calibrated (ours) & __A1_AUC__ & __A1_TPR1__ & __A1_TPR01__ \\
A2 subspace (ours) & __A2_AUC__ & __A2_TPR1__ & __A2_TPR01__ \\
A1 $+$ A2 fusion & __FUSE_AUC__ & __FUSE_TPR1__ & __FUSE_TPR01__ \\
\midrule
\emph{Merged full model (control)} & __MERGED_AUC__ & __MERGED_TPR1__ & __MERGED_TPR01__ \\
\bottomrule
\end{tabular}
"""

TABLE_RANK = r"""
\begin{tabular}{lccccc}
\toprule
$r$ & Params (M) & Macro-F1 & MIA AUC & Exposure & Edit dist.\ $\downarrow$ \\
\midrule
1 & __R1_PARAMS__ & __R1_F1__ & __R1_AUC__ & __R1_EXP__ & __R1_EDIT__ \\
2 & __R2_PARAMS__ & __R2_F1__ & __R2_AUC__ & __R2_EXP__ & __R2_EDIT__ \\
4 & __R4_PARAMS__ & __R4_F1__ & __R4_AUC__ & __R4_EXP__ & __R4_EDIT__ \\
8 & __R8_PARAMS__ & __R8_F1__ & __R8_AUC__ & __R8_EXP__ & __R8_EDIT__ \\
16 & __R16_PARAMS__ & __R16_F1__ & __R16_AUC__ & __R16_EXP__ & __R16_EDIT__ \\
64 & __R64_PARAMS__ & __R64_F1__ & __R64_AUC__ & __R64_EXP__ & __R64_EDIT__ \\
\bottomrule
\end{tabular}
"""

TABLE_RECON = r"""
\begin{tabular}{lcccc}
\toprule
Setting & Edit dist.\ $\downarrow$ & Exact spans & PHI recall & Exposure \\
\midrule
1 epoch, $|\D|{=}10^4$ & __RECON1_EDIT__ & __RECON1_EXACT__ & __RECON1_PHI__ & __RECON1_EXP__ \\
10 epochs, $|\D|{=}10^4$ & __RECON10_EDIT__ & __RECON10_EXACT__ & __RECON10_PHI__ & __RECON10_EXP__ \\
10 epochs, $|\D|{=}10^2$ & __RECON100_EDIT__ & __RECON100_EXACT__ & __RECON100_PHI__ & __RECON100_EXP__ \\
Canary $\times1$ & --- & --- & --- & __CANARY1_EXP__ \\
Canary $\times16$ & --- & --- & --- & __CANARY16_EXP__ \\
\bottomrule
\end{tabular}
"""

TABLE_DEF = r"""
\begin{tabular}{lcccc}
\toprule
Defense & Formal guar. & Macro-F1 & MIA AUC & PHI recall $\downarrow$ \\
\midrule
None & --- & __DEF_NONE_F1__ & __DEF_NONE_AUC__ & __DEF_NONE_PHI__ \\
D2 truncation ($r'{=}2$) & --- & __DEF_D2_F1__ & __DEF_D2_AUC__ & __DEF_D2_PHI__ \\
D3 frozen-$A$ & --- & __DEF_D3_F1__ & __DEF_D3_AUC__ & __DEF_D3_PHI__ \\
D4 merge ($k{=}8$, TIES) & --- & __DEF_D4_F1__ & __DEF_D4_AUC__ & __DEF_D4_PHI__ \\
D5 \textsc{AdaptGuard} & partial & __DEF_D5_F1__ & __DEF_D5_AUC__ & __DEF_D5_PHI__ \\
D1 DP-LoRA $\varepsilon{=}8$ & $(8,\delta)$ & __DEF_DP8_F1__ & __DEF_DP8_AUC__ & __DEF_DP8_PHI__ \\
D1 DP-LoRA $\varepsilon{=}1$ & $(1,\delta)$ & __DEF_DP1_F1__ & __DEF_DP1_AUC__ & __DEF_DP1_PHI__ \\
\bottomrule
\end{tabular}
"""


def apply_placeholders(template: str, rep: Dict[str, str]) -> str:
    out = template
    for key, val in rep.items():
        out = out.replace(f"__{key}__", val)
    # leave unresolved as XX.X
    out = re.sub(r"__[A-Z0-9_]+__", r"\\tbd", out)
    return out


def patch_tex(tex_path: str, rep: Dict[str, str], dry_run: bool = False) -> None:
    with open(tex_path) as handle:
        text = handle.read()

    # Capacity / abstract-ish substitutions where we use explicit markers later.
    # Also replace \tbd blocks inside known tables by rewriting whole tabulars.
    def replace_tabular_after(caption_label: str, new_tabular: str, src: str) -> str:
        # Find \label{caption_label} then the following tabular
        pat = rf"(\\label\{{{re.escape(caption_label)}\}}\s*)\\begin\{{tabular\}}.*?\\end\{{tabular\}}"
        def _repl(match):
            return match.group(1) + new_tabular.strip()
        new_src, n = re.subn(pat, _repl, src, count=1, flags=re.S)
        if n == 0:
            print(f"WARNING: could not patch table {caption_label}")
            return src
        return new_src

    text = replace_tabular_after("tab:mia", apply_placeholders(TABLE_MIA, rep), text)
    text = replace_tabular_after("tab:rank", apply_placeholders(TABLE_RANK, rep), text)
    text = replace_tabular_after("tab:recon", apply_placeholders(TABLE_RECON, rep), text)
    text = replace_tabular_after("tab:defense", apply_placeholders(TABLE_DEF, rep), text)

    # Capacity proposition numbers
    text = text.replace(
        r"holds $\sim$\tbd\,M parameters",
        "holds $\\sim$%s\\,M parameters" % rep.get("CAP_PARAMS_M", "13.6"),
    )
    text = text.replace(
        r"$r{=}16$, bf16 gives $\approx$\tbd\,MB against $\approx$1.5\,KB per"
        "\nnote, a ratio of $\\sim$\\tbd$\\times$.",
        "$r{=}16$, bf16 gives $\\approx$%s\\,MB against $\\approx$1.5\\,KB per"
        "\nnote, a ratio of $\\sim$%s$\\times$."
        % (rep.get("CAP_MB", "27.3"), rep.get("CAP_RATIO", "17749")),
    )
    # Abstract fills when available
    if "ABS_TPR01" in rep:
        text = text.replace(
            r"at \tbd\ TPR at $0.1\%$ FPR",
            "at %s TPR at $0.1\\%%$ FPR" % rep["ABS_TPR01"],
        )
    if "ABS_RANK_DELTA" in rep:
        text = text.replace(
            r"changes attack success by only \tbd\ points",
            "changes attack success by only %s points" % rep["ABS_RANK_DELTA"],
        )
    if "ABS_D5_RECON" in rep and "ABS_D5_F1DROP" in rep and "ABS_DP8_F1DROP" in rep:
        text = text.replace(
            "It cuts\nreconstruction to \\tbd\\ at \\tbd\\ points of macro-F1, versus \\tbd\\ points\n"
            "for DP-LoRA at $\\varepsilon{=}8$.",
            "It cuts\nreconstruction to %s at %s points of macro-F1, versus %s points\n"
            "for DP-LoRA at $\\varepsilon{=}8$."
            % (rep["ABS_D5_RECON"], rep["ABS_D5_F1DROP"], rep["ABS_DP8_F1DROP"]),
        )
    if "DEF_NONE_PHI" in rep:
        text = text.replace(
            r"recovers \tbd\% of",
            "recovers %s of" % rep["DEF_NONE_PHI"],
        )

    if dry_run:
        print("Dry run; not writing.")
        print("Keys:", sorted(rep.keys()))
        return
    with open(tex_path, "w") as handle:
        handle.write(text)
    print(f"Patched {tex_path} with {len(rep)} values")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="privacy_lora_artifacts")
    parser.add_argument("--tex", default="iclr2027_privacy_lora.tex")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--print_rep", action="store_true")
    args = parser.parse_args()
    rep = build_replacements(args.root)
    if args.print_rep:
        print(json.dumps(rep, indent=2))
    patch_tex(args.tex, rep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
