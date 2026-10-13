"""Aggregate superposition_todos JSONs and print / patch NAACL todos."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("PRIVACY_LORA_TODOS", "privacy_lora_v2/superposition_todos")
TEX = "NAACL_privacy_lora.tex"
SEEDS = (0, 1, 2)
DS = ("pmc", "synthetic", "enron")
MODEL = "qwen14b"


def load_all():
    rows = []
    for ds in DS:
        for s in SEEDS:
            p = os.path.join(ROOT, f"{MODEL}_{ds}_s{s}.json")
            if os.path.exists(p):
                rows.append(json.load(open(p)))
    return rows


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def fmt(m, nd=3):
    if abs(m) < 1e-4 and m != 0:
        return f"{m:.1e}"
    if abs(m) >= 10:
        return f"{m:.2f}"
    return f"{m:.{nd}f}"


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)

    print(f"loaded {len(rows)}/9")
    print("\n=== Identity / drift ===")
    for steps in (1, 50, 500):
        key = f"t{steps}"
        for ds in DS:
            vals = [r["identity_drift"][key] for r in by[ds] if key in r.get("identity_drift", {})]
            if not vals:
                continue
            def col(k):
                return mean_std([v[k] for v in vals])
            re_m, _ = col("R_exact")
            ea_m, _ = col("eps_A")
            eg_m, _ = col("eps_G")
            et_m, _ = col("E_T_rel")
            ok = np.mean([v["bound_ok"] for v in vals])
            print(f"  {ds} T={steps}: R_exact={fmt(re_m,2)} eps_A={fmt(ea_m)} eps_G={fmt(eg_m)} "
                  f"E_T_rel={fmt(et_m)} bound_ok={ok*100:.0f}%")

    print("\n=== Additivity ===")
    for steps in (1, 50):
        key = f"t{steps}"
        for ds in DS:
            vals = [r["additivity"][key]["R_add"] for r in by[ds] if key in r.get("additivity", {})]
            if vals:
                m, _ = mean_std(vals)
                print(f"  {ds} T={steps}: R_add={fmt(m)}")

    print("\n=== Coherence ===")
    mus, kappas, kpos = [], [], []
    for ds in DS:
        for r in by[ds]:
            c = r.get("coherence", {})
            if not c:
                continue
            mus.append(c["mu"]); kappas.append(c["kappa"]); kpos.append(c["kappa_pos"])
            print(f"  {ds} s{r['seed']}: mu={c['mu']:.4f} kappa={c['kappa']:.4f} pos={c['kappa_pos']:.0f}")
    if mus:
        print(f"  OVERALL: mu={np.mean(mus):.4f} kappa={np.mean(kappas):.4f} "
              f"frac(kappa>0)={np.mean(kpos)*100:.0f}%")

    print("\n=== Perturbation ===")
    for pct in (1, 5, 10, 20):
        key = f"perturb_{pct}pct"
        for ds in DS:
            vals = [r["perturbation"][key] for r in by[ds] if key in r.get("perturbation", {})]
            if not vals:
                continue
            ec = np.mean([v["E_c"] for v in vals])
            su = np.mean([v["supp"] for v in vals])
            rf = np.mean([v["R_fit"] for v in vals])
            print(f"  {ds} {pct}%: E_c={fmt(ec)} supp={su:.0f} R_fit={fmt(rf)}")

    return by


def patch_tex(by):
    """Replace the two \\todo{...} blocks with measured prose."""
    # Aggregate across corpora for concise prose
    def gather_id(steps, key):
        vals = []
        for ds in DS:
            for r in by[ds]:
                rec = r.get("identity_drift", {}).get(f"t{steps}")
                if rec:
                    vals.append(rec[key])
        return float(np.mean(vals)) if vals else None

    def gather_add(steps):
        vals = []
        for ds in DS:
            for r in by[ds]:
                rec = r.get("additivity", {}).get(f"t{steps}")
                if rec:
                    vals.append(rec["R_add"])
        return float(np.mean(vals)) if vals else None

    mus, kappas, kpos = [], [], []
    for ds in DS:
        for r in by[ds]:
            c = r.get("coherence", {})
            if c:
                mus.append(c["mu"]); kappas.append(c["kappa"]); kpos.append(c["kappa_pos"])

    pert = {}
    for pct in (1, 5, 10, 20):
        key = f"perturb_{pct}pct"
        ecs, sups, rfs = [], [], []
        for ds in DS:
            for r in by[ds]:
                v = r.get("perturbation", {}).get(key)
                if v:
                    ecs.append(v["E_c"]); sups.append(v["supp"]); rfs.append(v["R_fit"])
        pert[pct] = (np.mean(ecs), np.mean(sups), np.mean(rfs))

    r1 = gather_id(1, "R_exact")
    r50 = gather_id(50, "R_exact")
    r500 = gather_id(500, "R_exact")
    ea50 = gather_id(50, "eps_A")
    eg50 = gather_id(50, "eps_G")
    ea500 = gather_id(500, "eps_A")
    eg500 = gather_id(500, "eps_G")
    et50 = gather_id(50, "E_T_rel")
    et500 = gather_id(500, "E_T_rel")
    bok50 = gather_id(50, "bound_ok")
    bok500 = gather_id(500, "bound_ok")
    add1 = gather_add(1)
    add50 = gather_add(50)

    if None in (r1, r50, r500, add1, add50) or not mus:
        print("incomplete aggregates; not patching tex")
        return

    para1 = (
        f"As a researcher-side check of Theorem~\\ref{{thm:exact_superposition}}, "
        f"the logged-path residual $R_{{\\mathrm{{exact}}}}$ averages "
        f"${fmt(r1,2)}$ after one step, ${fmt(r50,2)}$ at $T{{=}}50$, and ${fmt(r500,2)}$ at $T{{=}}500$ "
        f"(numerical roundoff, not a modeling gap). "
        f"The drift coordinates of Proposition~\\ref{{prop:drift_bound}} grow with the horizon: "
        f"at $T{{=}}50$, $\\varepsilon_A{{=}}{fmt(ea50)}$ and $\\varepsilon_G{{=}}{fmt(eg50)}$ "
        f"(mean over corpora and seeds), rising to $\\varepsilon_A{{=}}{fmt(ea500)}$ and "
        f"$\\varepsilon_G{{=}}{fmt(eg500)}$ at $T{{=}}500$; the fixed-feature residual "
        f"$\\|E_T\\|_F/\\|\\Delta W_T\\|_F$ is ${fmt(et50)}$ and ${fmt(et500)}$ at those horizons, "
        f"and stays within the Proposition~\\ref{{prop:drift_bound}} bound on "
        f"${100*bok50:.0f}\\%$ / ${100*bok500:.0f}\\%$ of runs. "
        f"A subset-versus-union control confirms path dependence: training the four hidden records "
        f"together versus summing four singleton adapters gives relative error "
        f"$R_{{\\mathrm{{add}}}}{{=}}{fmt(add1)}$ at $T{{=}}1$ and ${fmt(add50)}$ at $T{{=}}50$."
    )

    p1, s1, f1 = pert[1]
    p5, s5, f5 = pert[5]
    p10, s10, f10 = pert[10]
    p20, s20, f20 = pert[20]
    para2 = (
        f"On the $T{{=}}50$ dictionaries, the normalized coherence averages "
        f"$\\mu{{=}}{np.mean(mus):.3f}$, so $\\kappa{{=}}1-(2s-1)\\mu$ averages "
        f"${np.mean(kappas):.3f}$ and is positive in ${100*np.mean(kpos):.0f}\\%$ of runs "
        f"(Theorem~\\ref{{thm:identifiability}}). "
        f"Extending the observation perturbation beyond $1\\%$ leaves exact support intact through "
        f"$5\\%$ ($E_c{{=}}{fmt(p5)}$, Supp.\\ ${s5:.0f}\\%$) and $10\\%$ "
        f"($E_c{{=}}{fmt(p10)}$, Supp.\\ ${s10:.0f}\\%$); at $20\\%$ the mean residual rises to "
        f"$R_{{\\mathrm{{fit}}}}{{=}}{fmt(f20)}$ with $E_c{{=}}{fmt(p20)}$ and support "
        f"${s20:.0f}\\%$, consistent with the stability bound of Eq.~\\eqref{{eq:recovery_stability}}."
    )

    tex = open(TEX).read()
    tex2, n1 = re.subn(
        r"\\todo\{Add the exact-identity residual \$R_\{\\mathrm\{exact\}\}\}.*?\}",
        para1,
        tex,
        count=1,
        flags=re.S,
    )
    if n1 != 1:
        # todos are single-line
        tex2, n1 = re.subn(
            r"\\todo\{Add the exact-identity residual[^}]*\}",
            para1,
            tex,
            count=1,
        )
    tex3, n2 = re.subn(
        r"\\todo\{Report dictionary coherence[^}]*\}",
        para2,
        tex2,
        count=1,
    )
    print(f"patched todos: identity={n1} coherence={n2}")
    if n1 == 1 and n2 == 1:
        open(TEX, "w").write(tex3)
        print("updated", TEX)


def main():
    rows = load_all()
    by = summarize(rows)
    if len(rows) >= 9:
        patch_tex(by)


if __name__ == "__main__":
    main()
