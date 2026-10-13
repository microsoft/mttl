"""Run one experimental arm: build corpus, train, audit at checkpoints, write JSON.

Usage:  python -m projects.privacy_lora.v2.run_arm --spec '{"name": "...", ...}'
        python -m projects.privacy_lora.v2.run_arm --spec_file arms.jsonl [--only NAME]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from projects.privacy_lora.v2 import attacks as A
from projects.privacy_lora.v2.data import CorpusSpec, Example, cohort_stats, get_corpus, make_examples
from projects.privacy_lora.v2.defenses import distill_adaptguard
from projects.privacy_lora.v2.modeling import (attach_method, load_base_model, load_lora_state, lora_modules, lora_state,
                                              merge_lora_states, set_lora_scaling, truncate_lora_rank)
from projects.privacy_lora.v2.train import TrainSpec, train_model

RESULTS = os.environ.get("PRIVACY_LORA_RESULTS", "privacy_lora_v2/results")
CACHE = os.environ.get("PRIVACY_LORA_CACHE", "privacy_lora_v2/cache")


@dataclass
class Arm:
    name: str
    # data
    dataset: str = "synthetic"
    data_seed: int = 0
    n_train: int = 1000
    n_holdout: int = 5000
    n_public: int = 3000
    female_frac: float = 0.5
    dx_skew: float = 0.0
    n_members_eval: int = 1000
    n_nonmembers_eval: int = 5000
    # model / method
    backbone: str = "meta-llama/Llama-3.1-8B-Instruct"
    method: str = "lora"          # lora | full | ia3 | head
    rank: int = 16
    alpha: Optional[float] = None
    targets: str = "attn"         # attn | mlp | all
    variant: str = "lora"         # lora | rslora
    modules_regex: str = ".*"
    freeze_a: bool = False        # D3
    lora_dropout: float = 0.05
    # training
    fmt: str = "lm"               # lm | inst
    epochs: int = 3
    eval_epochs: Sequence[int] = field(default_factory=lambda: [1, 2, 3])
    full_eval_epochs: Optional[Sequence[int]] = None  # epochs with A2 / exposure / B-side; default = eval_epochs
    lr: float = 1e-4
    batch: int = 8
    micro_batch: int = 4
    optimizer: str = "adamw"
    max_length: Optional[int] = None
    seed: int = 0
    max_steps: Optional[int] = None
    # defenses
    defense: str = "none"         # none | d2 | d4 | d5 | dp
    d2_ranks: Sequence[int] = field(default_factory=lambda: [2, 4, 8])
    d4_shards: int = 8
    d5_public: str = "in"         # in | shift
    d5_n_public: int = 1000
    d5_top_k: int = 50
    d5_noise: float = 0.5
    d5_steps: int = 600
    d5_pub_only: bool = False
    dp_epsilon: Optional[float] = None
    dp_batch: int = 256
    dp_clip: float = 1.0
    # attacks
    a2_variants: Sequence[str] = field(default_factory=lambda: ["A", "A-A0"])
    b_side: bool = False
    b_side_n: Sequence[int] = field(default_factory=lambda: [300, 1000])
    exposure: bool = True
    exposure_samples: int = 2000
    recon: str = "none"           # none | greedy | gcg
    recon_restarts: int = 4
    recon_steps: int = 40
    a5: bool = False
    dup_probe: bool = True
    memorize: bool = False        # E17: single-note memorisation demo
    n_boot: int = 1000
    save_scores: bool = True
    notes: str = ""


def _short(name: str) -> str:
    return name.split("/")[-1].replace(".", "_")


def _npz_key(arm: Arm, corpus_key: str, max_length: int) -> str:
    return f"{_short(arm.backbone)}__{corpus_key}__{arm.fmt}__L{max_length}"


def base_scores(model, tok, arm: Arm, corpus: Dict, sets: Dict[str, List[Example]], max_length: int, device: str) -> Dict:
    """Base-model (W0) statistics, cached per (backbone, corpus, format)."""
    os.makedirs(CACHE, exist_ok=True)
    key = _npz_key(arm, corpus["key"], max_length)
    path = os.path.join(CACHE, key + ".npz")
    jpath = os.path.join(CACHE, key + ".json")
    out: Dict = {}
    extra: Dict = {}
    if os.path.exists(path) and os.path.exists(jpath):
        blob = np.load(path)
        cached = {"members": {}, "nonmembers": {}}
        for k in blob.files:
            grp, stat = k.split("::")
            cached[grp][stat] = blob[k]
        if len(cached["members"]["loss"]) >= len(sets["members"]) and len(cached["nonmembers"]["loss"]) >= len(sets["nonmembers"]):
            out["members"] = {k: v[: len(sets["members"])] for k, v in cached["members"].items()}
            out["nonmembers"] = {k: v[: len(sets["nonmembers"])] for k, v in cached["nonmembers"].items()}
        with open(jpath) as f:
            extra = json.load(f)
    if "members" not in out:
        out = {"members": A.score_examples(model, tok, sets["members"], max_length, device, desc="base-m"),
               "nonmembers": A.score_examples(model, tok, sets["nonmembers"], max_length, device, desc="base-n")}
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez(tmp, **{f"{g}::{k}": v for g in ("members", "nonmembers") for k, v in out[g].items()})
        os.replace(tmp, path)
    changed = False
    if "utility" not in extra:
        extra["utility"] = A.utility(model, tok, sets["val"], max_length, device)
        changed = True
    if arm.exposure and "exposure" not in extra:
        extra["exposure"] = [A.canary_exposure_v2(model, tok, c, device, n_samples=arm.exposure_samples, seed=i)
                             for i, c in enumerate(corpus["canaries"])]
        extra["exposure_decoys"] = [A.canary_exposure_v2(model, tok, c, device, n_samples=arm.exposure_samples, seed=100 + i)
                                    for i, c in enumerate(corpus["decoys"])]
        changed = True
    if arm.a5 and arm.dataset == "synthetic" and "a5" not in extra:
        extra["a5"] = A.property_inference(model, tok, device, seed=arm.seed)
        changed = True
    if arm.dup_probe and corpus["dup_notes"] and "dup_probe" not in extra:
        extra["dup_probe"] = [A.verbatim_probe(model, tok, d["text"], device) for d in corpus["dup_notes"]]
        changed = True
    if changed:
        tmp = f"{jpath}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(extra, f)
        os.replace(tmp, jpath)
    out.update(extra)
    return out


class Auditor:
    def __init__(self, arm: Arm, corpus: Dict, sets: Dict[str, List[Example]], base: Dict, tok, device: str, max_length: int, out_dir: str):
        self.arm, self.corpus, self.sets, self.base, self.tok, self.device, self.max_length = arm, corpus, sets, base, tok, device, max_length
        self.out_dir = out_dir
        self.evals: Dict[str, Dict] = {}
        self.scores: Dict[str, Dict[str, np.ndarray]] = {}

    def run(self, model, tag: str, full: bool = True, mia: bool = True) -> Dict:
        arm, s = self.arm, self.sets
        t0 = time.time()
        res: Dict = {"tag": tag}
        model.eval()
        res["utility"] = A.utility(model, self.tok, s["val"], self.max_length, self.device)
        has_lora = len(lora_modules(model)) > 0
        if mia:
            am = A.score_examples(model, self.tok, s["members"], self.max_length, self.device, desc=f"{tag}-m")
            an = A.score_examples(model, self.tok, s["nonmembers"], self.max_length, self.device, desc=f"{tag}-n")
            sm = A.attack_scores(am, self.base["members"])
            sn = A.attack_scores(an, self.base["nonmembers"])
            if full and has_lora and arm.a2_variants:
                em = A.a2_energies(model, self.tok, s["members"], self.max_length, self.device, arm.a2_variants, desc=f"{tag}-a2m")
                en = A.a2_energies(model, self.tok, s["nonmembers"], self.max_length, self.device, arm.a2_variants, desc=f"{tag}-a2n")
                for v in em:
                    sm[f"a2_{v}"], sn[f"a2_{v}"] = em[v], en[v]
                if "A" in em:
                    sm["fusion"], sn["fusion"] = A.zfuse((sm["ref"], sn["ref"]), (em["A"], en["A"]))
                    sm["fusion_mink"], sn["fusion_mink"] = A.zfuse((sm["ref_mink"], sn["ref_mink"]), (em["A"], en["A"]))
                if arm.b_side:
                    nm, nn_ = arm.b_side_n
                    bm = A.b_side_energies(model, self.tok, s["members"][:nm], self.max_length, self.device)
                    bn = A.b_side_energies(model, self.tok, s["nonmembers"][:nn_], self.max_length, self.device)
                    sm["a2_B"], sn["a2_B"] = bm, bn
            res["mia"] = {k: A.mia_metrics(sm[k], sn[k], n_boot=arm.n_boot, seed=arm.seed) for k in sm}
            res["adapted_stats"] = {"member_loss": float(am["loss"].mean()), "nonmember_loss": float(an["loss"].mean()),
                                    "member_ntok": float(am["n_tok"].mean()), "nonmember_ntok": float(an["n_tok"].mean())}
            if arm.save_scores:
                self.scores[tag] = {**{f"m::{k}": v for k, v in sm.items()}, **{f"n::{k}": v for k, v in sn.items()}}
        if full and arm.exposure:
            res["exposure"] = [A.canary_exposure_v2(model, self.tok, c, self.device, n_samples=arm.exposure_samples, seed=i)
                               for i, c in enumerate(self.corpus["canaries"])]
            res["exposure_decoys"] = [A.canary_exposure_v2(model, self.tok, c, self.device, n_samples=arm.exposure_samples, seed=100 + i)
                                      for i, c in enumerate(self.corpus["decoys"])]
        if full and arm.dup_probe and self.corpus["dup_notes"]:
            res["dup_probe"] = [{"reps": d["reps"], **A.verbatim_probe(model, self.tok, d["text"], self.device)} for d in self.corpus["dup_notes"]]
            res["dup_probe_control"] = [A.verbatim_probe(model, self.tok, t, self.device) for t in self.corpus["nonmembers"][:3]]
            res["dup_probe_members"] = [A.verbatim_probe(model, self.tok, t, self.device) for t in self.corpus["members"][:3]]
        res["seconds"] = time.time() - t0
        self.evals[tag] = res
        return res

    def recon(self, model, tag: str = "final") -> Dict:
        arm = self.arm
        pub_first = self.corpus["public"][0].split("\n")[0] if self.corpus["public"] else "Discharge Summary"
        if arm.dataset == "enron":
            inits = ["From:", "Subject: RE:", "Please contact me at", pub_first]
        else:
            inits = ["Discharge Summary\nPatient Name:", "CONFIDENTIAL PHI MARKER:", "Patient Name:", pub_first + "\nPatient Name:", "MRN:"]
        inits = inits[: arm.recon_restarts]
        configs = {"greedy": None}
        if arm.recon == "gcg" and lora_modules(model):
            configs.update({"gcg_lm": dict(lam=1.0, mu=0.0), "gcg_sub": dict(lam=0.0, mu=1.0), "gcg_full": dict(lam=1.0, mu=1.0)})
        out = {}
        model.eval()
        allowed = None
        if len(configs) > 1:
            allowed = A.public_vocab_mask(self.tok, self.corpus["public"], model.get_input_embeddings().weight.shape[0], self.device)
        for cname, cfg in configs.items():
            gens, prefixes, objs = [], [], []
            for i, init in enumerate(inits):
                if cfg is None:
                    prefix_text = init
                else:
                    r = A.gcg_invert(model, self.tok, self.device, init, n_steps=arm.recon_steps, seed=i, allowed_mask=allowed, **cfg)
                    prefix_text = r["prefix_text"]
                    objs.append(r["objective"])
                prefixes.append(prefix_text)
                gens += A.generate(model, self.tok, [prefix_text], self.device, max_new_tokens=128)
                gens += A.generate(model, self.tok, [prefix_text] * 8, self.device, max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.95)
            m = A.extraction_metrics(gens, self.corpus["canaries"], self.corpus["train_all"], self.corpus["nonmembers"], self.corpus["decoys"])
            m["prefixes"] = prefixes
            m["objective_curves"] = objs
            m["example"] = gens[0][:400]
            out[cname] = m
        self.evals.setdefault(tag, {})["recon"] = out
        return out

    def a5(self, model, tag: str = "final") -> Dict:
        est = A.property_inference(model, self.tok, self.device, seed=self.arm.seed)
        truth = cohort_stats(self.corpus["member_attrs"])
        res = {"estimate": est, "truth": truth}
        if est["female_frac"] is not None:
            res["female_abs_err"] = abs(est["female_frac"] - truth["female_frac"])
        res["dx_tv"] = A.tv_distance(est["dx_dist"], truth["dx_dist"]) if est["dx_dist"] else None
        base = self.base.get("a5")
        if base:
            res["base"] = base
            res["base_female_abs_err"] = abs(base["female_frac"] - truth["female_frac"]) if base.get("female_frac") is not None else None
            res["base_dx_tv"] = A.tv_distance(base["dx_dist"], truth["dx_dist"]) if base.get("dx_dist") else None
        self.evals.setdefault(tag, {})["a5"] = res
        return res

    def save(self, extra: Dict):
        os.makedirs(self.out_dir, exist_ok=True)
        blob = {"arm": asdict(self.arm), "evals": self.evals, "base": {k: v for k, v in self.base.items() if k not in ("members", "nonmembers")}, **extra}
        tmp = os.path.join(self.out_dir, self.arm.name + ".json.tmp")
        with open(tmp, "w") as f:
            json.dump(blob, f, indent=1, default=float)
        os.replace(tmp, os.path.join(self.out_dir, self.arm.name + ".json"))
        if self.scores:
            np.savez_compressed(os.path.join(self.out_dir, self.arm.name + "_scores.npz"),
                                **{f"{tag}::{k}": v for tag, d in self.scores.items() for k, v in d.items()})


def run(arm: Arm, device: str = "cuda") -> Dict:
    t_start = time.time()
    out_path = os.path.join(RESULTS, arm.name + ".json")
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        if not prev.get("partial", False):
            print(f"[skip] {out_path} exists")
            return prev
    torch.manual_seed(arm.seed)
    random.seed(arm.seed)
    np.random.seed(arm.seed)
    max_length = arm.max_length or (512 if arm.fmt == "inst" else 384)

    spec = CorpusSpec(dataset=arm.dataset, n_train=arm.n_train, n_holdout=arm.n_holdout, n_public=arm.n_public, seed=arm.data_seed,
                      female_frac=arm.female_frac, dx_skew=arm.dx_skew)
    corpus = get_corpus(spec)
    ip = corpus["inst_prompt"]
    sets = {
        "train": make_examples(corpus["train_all"], arm.fmt, ip),
        "members": make_examples(corpus["members"][: arm.n_members_eval], arm.fmt, ip),
        "nonmembers": make_examples(corpus["nonmembers"][: arm.n_nonmembers_eval], arm.fmt, ip),
        "val": make_examples(corpus["val"], arm.fmt, ip),
    }
    if arm.memorize:
        sets["train"] = make_examples([corpus["dup_notes"][0]["text"]], arm.fmt, ip)

    model, tok = load_base_model(arm.backbone, device)
    base = base_scores(model, tok, arm, corpus, sets, max_length, device)
    auditor = Auditor(arm, corpus, sets, base, tok, device, max_length, RESULTS)

    info = attach_method(model, arm.method, rank=arm.rank, alpha=arm.alpha, targets=arm.targets, modules_regex=arm.modules_regex,
                         freeze_a=arm.freeze_a, lora_dropout=arm.lora_dropout, variant=arm.variant)
    optimizer = arm.optimizer
    if arm.method == "full" and optimizer == "adamw":
        optimizer = "adamw8bit"
    tspec = TrainSpec(epochs=arm.epochs, lr=arm.lr, batch=arm.batch, micro_batch=arm.micro_batch if arm.method != "full" else 1,
                      optimizer=optimizer, max_length=max_length, seed=arm.seed, eval_epochs=list(arm.eval_epochs), max_steps=arm.max_steps,
                      dp_epsilon=arm.dp_epsilon if arm.defense == "dp" else None, dp_batch=arm.dp_batch, dp_clip=arm.dp_clip)
    full_epochs = set(arm.full_eval_epochs if arm.full_eval_epochs is not None else arm.eval_epochs)
    train_info: Dict = {}

    def on_epoch_end(epoch: int, hist: Dict):
        if arm.defense in ("d4", "d5"):
            return  # audited after merging / distillation
        tag = f"ep{epoch}"
        r = auditor.run(model, tag, full=epoch in full_epochs, mia=not arm.memorize)
        r["train"] = hist
        auditor.save({"train": train_info, "method": info, "corpus_key": corpus["key"], "partial": True})
        model.train()

    if arm.defense == "d4":
        k = arm.d4_shards
        train_ex = sets["train"]
        shards = [train_ex[i::k] for i in range(k)]
        init = lora_state(model)
        states = []
        for si, shard in enumerate(shards):
            torch.manual_seed(arm.seed * 100 + si)
            for _, m in lora_modules(model):
                m.reset_parameters()
            ti = train_model(model, tok, shard, dataclasses.replace(tspec, seed=arm.seed * 100 + si, eval_epochs=[]), device)
            states.append(lora_state(model))
            train_info[f"shard{si}"] = {"steps": ti["history"][-1]["step"], "loss": ti["history"][-1]["train_loss"]}
        for method in ("ties", "uniform"):
            load_lora_state(model, merge_lora_states(states, method=method))
            set_lora_scaling(model, 1.0)
            auditor.run(model, f"d4_{method}", full=True)
        load_lora_state(model, merge_lora_states(states, method="ties"))
    elif arm.defense == "d5":
        train_info["teacher"] = train_model(model, tok, sets["train"], tspec, device, val_examples=sets["val"])
        auditor.run(model, "teacher", full=True)
        if arm.recon != "none":
            auditor.recon(model, "teacher")
        teacher = model
        student, _ = load_base_model(arm.backbone, device)
        attach_method(student, "lora", rank=arm.rank, alpha=arm.alpha, targets=arm.targets, lora_dropout=arm.lora_dropout)
        pub_texts = corpus["public_shift"] if arm.d5_public == "shift" else corpus["public"]
        pub = make_examples(pub_texts[: arm.d5_n_public], arm.fmt, ip)
        train_info["d5"] = distill_adaptguard(teacher if not arm.d5_pub_only else None, student, tok, pub, device, max_length,
                                              n_steps=arm.d5_steps, micro_batch=arm.micro_batch, lr=arm.lr, top_k=arm.d5_top_k,
                                              logit_noise=arm.d5_noise, pub_only=arm.d5_pub_only, seed=arm.seed)
        del teacher
        torch.cuda.empty_cache()
        model = student
        auditor.run(model, "final", full=True)
    else:
        train_info["train"] = train_model(model, tok, sets["train"], tspec, device, on_epoch_end=on_epoch_end, val_examples=sets["val"])
        if arm.epochs not in set(arm.eval_epochs) and not arm.max_steps:
            r = auditor.run(model, f"ep{arm.epochs}", full=True, mia=not arm.memorize)
        # early-stopping epoch = argmin val NLL over the history
        hist = train_info["train"]["history"]
        vals = [(h["val_nll"], h["epoch"]) for h in hist if h.get("val_nll") is not None]
        if vals:
            train_info["early_stop_epoch"] = min(vals)[1]

    final_tag = f"ep{arm.epochs}" if arm.defense not in ("d4", "d5") else ("final" if arm.defense == "d5" else "d4_ties")
    if arm.memorize:
        auditor.evals.setdefault(final_tag, {})
    if arm.recon != "none" and not arm.memorize:
        auditor.recon(model, final_tag)
    if arm.a5 and arm.dataset == "synthetic":
        auditor.a5(model, final_tag)
    if arm.defense == "d2":
        orig = lora_state(model)
        for r in arm.d2_ranks:
            load_lora_state(model, orig)
            set_lora_scaling(model, (arm.alpha or arm.rank) / arm.rank)
            tinfo = truncate_lora_rank(model, r)
            res = auditor.run(model, f"d2_r{r}", full=True)
            res["truncation"] = tinfo
            if arm.recon != "none":
                auditor.recon(model, f"d2_r{r}")
        load_lora_state(model, orig)
        set_lora_scaling(model, (arm.alpha or arm.rank) / arm.rank)

    train_info["gpu_hours"] = (time.time() - t_start) / 3600.0
    auditor.save({"train": train_info, "method": info, "corpus_key": corpus["key"], "max_length": max_length,
                  "n_train_all": len(sets["train"]), "partial": False})
    print(f"[done] {arm.name} in {train_info['gpu_hours']:.2f} GPU-h")
    return {"ok": True}


def parse_arm(d: Dict) -> Arm:
    fields = {f.name for f in dataclasses.fields(Arm)}
    unknown = set(d) - fields
    if unknown:
        raise ValueError(f"unknown arm fields: {unknown}")
    return Arm(**d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=str, default=None)
    ap.add_argument("--spec_file", type=str, default=None)
    ap.add_argument("--only", type=str, default=None, help="comma-separated arm names")
    args = ap.parse_args()
    arms: List[Arm] = []
    if args.spec:
        arms.append(parse_arm(json.loads(args.spec)))
    if args.spec_file:
        with open(args.spec_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    arms.append(parse_arm(json.loads(line)))
    if args.only:
        keep = set(args.only.split(","))
        arms = [a for a in arms if a.name in keep]
    for arm in arms:
        try:
            run(arm)
        except Exception as e:  # keep going with the remaining arms
            import traceback

            traceback.print_exc()
            os.makedirs(RESULTS, exist_ok=True)
            with open(os.path.join(RESULTS, arm.name + ".failed"), "w") as f:
                f.write(f"{type(e).__name__}: {e}\n")
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
