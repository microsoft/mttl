"""Scoring, membership-inference metrics, exposure, extraction and property attacks (v2)."""

from __future__ import annotations

import math
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from projects.privacy_lora.clinical_data import DIAGNOSES
from projects.privacy_lora.v2.data import Example, zlib_bytes
from projects.privacy_lora.v2.modeling import lora_modules, set_lora_enabled


# --------------------------------------------------------------------------- #
# Tokenisation helpers
# --------------------------------------------------------------------------- #
def encode_example(tok, ex: Example, max_length: int) -> Tuple[List[int], int]:
    ids = tok(ex.text, truncation=True, max_length=max_length, add_special_tokens=True)["input_ids"]
    if ex.prompt:
        n_prompt = len(tok(ex.prompt, add_special_tokens=True)["input_ids"])
    else:
        n_prompt = 1
    n_prompt = int(min(n_prompt, max(len(ids) - 8, 1)))
    return ids, n_prompt


def collate(batch: List[Tuple[List[int], int]], pad_id: int, device: str):
    T = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), T), dtype=torch.long)
    labels = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, (ids, n_prompt) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
        labels[i, n_prompt : len(ids)] = torch.tensor(ids[n_prompt:])
    return input_ids.to(device), attn.to(device), labels.to(device)


# --------------------------------------------------------------------------- #
# Per-example likelihood statistics (loss, zlib, Min-K%, Min-K%++)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_examples(model, tok, examples: Sequence[Example], max_length: int, device: str,
                   batch_size: int = 8, k_frac: float = 0.2, desc: str = "score") -> Dict[str, np.ndarray]:
    model.eval()
    enc = [encode_example(tok, ex, max_length) for ex in examples]
    order = np.argsort([-len(e[0]) for e in enc])
    out = {k: np.zeros(len(examples)) for k in ("loss", "sum_lp", "n_tok", "mink", "minkpp", "zlib", "chars")}
    for start in tqdm(range(0, len(examples), batch_size), desc=desc, leave=False):
        idx = order[start : start + batch_size]
        input_ids, attn, labels = collate([enc[i] for i in idx], tok.pad_token_id, device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float()
        tgt = labels[:, 1:]
        mask = tgt != -100
        logprobs = torch.log_softmax(logits, dim=-1)
        tok_lp = logprobs.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        probs = logprobs.exp()
        mu = (probs * logprobs).sum(-1)
        sigma = ((probs * logprobs.pow(2)).sum(-1) - mu.pow(2)).clamp_min(1e-6).sqrt()
        z = (tok_lp - mu) / sigma
        for j, i in enumerate(idx):
            m = mask[j]
            lp = tok_lp[j][m]
            n = int(m.sum().item())
            if n == 0:
                continue
            kk = max(1, int(math.ceil(k_frac * n)))
            out["loss"][i] = -lp.mean().item()
            out["sum_lp"][i] = lp.sum().item()
            out["n_tok"][i] = n
            out["mink"][i] = torch.topk(lp, kk, largest=False).values.mean().item()
            out["minkpp"][i] = torch.topk(z[j][m], kk, largest=False).values.mean().item()
            out["zlib"][i] = zlib_bytes(examples[i].target)
            out["chars"][i] = len(examples[i].target)
        del logits, logprobs, probs
    return out


def attack_scores(adapted: Dict[str, np.ndarray], base: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Membership scores (higher = more member-like) for each attack family."""
    s = {
        "loss": -adapted["loss"],
        "zlib": -adapted["loss"] * adapted["n_tok"] / np.maximum(adapted["zlib"], 1),
        "mink": adapted["mink"],
        "minkpp": adapted["minkpp"],
        # blind baselines: do not use the adapter at all
        "blind_len": -adapted["n_tok"],
        "blind_zlib": -adapted["zlib"] / np.maximum(adapted["chars"], 1),
    }
    if base is not None:
        s["ref"] = base["loss"] - adapted["loss"]            # A1: log p_{W0+dW} - log p_{W0}, per token
        s["ref_mink"] = adapted["mink"] - base["mink"]
        s["blind_base_loss"] = -base["loss"]
    return s


# --------------------------------------------------------------------------- #
# MIA metrics with bootstrap CIs and log-log ROC
# --------------------------------------------------------------------------- #
FPR_GRID = np.logspace(-4, 0, 33)


def _auc(m: np.ndarray, n: np.ndarray) -> float:
    y = np.concatenate([m, n])
    ranks = np.empty(len(y))
    order = np.argsort(y, kind="mergesort")
    ranks[order] = np.arange(1, len(y) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.bincount(inv, weights=ranks)
        ranks = sums[inv] / counts[inv]
    return float((ranks[: len(m)].sum() - len(m) * (len(m) + 1) / 2) / (len(m) * len(n)))


def _tpr_at(m: np.ndarray, n: np.ndarray, fpr: float) -> float:
    thr = np.quantile(n, 1.0 - fpr, method="higher")
    return float((m > thr).mean())


def mia_metrics(m: np.ndarray, n: np.ndarray, n_boot: int = 1000, seed: int = 0) -> Dict:
    m = np.asarray(m, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    m = m[np.isfinite(m)]
    n = n[np.isfinite(n)]
    res = {
        "auc": _auc(m, n),
        "tpr_at_1pct_fpr": _tpr_at(m, n, 0.01),
        "tpr_at_0_1pct_fpr": _tpr_at(m, n, 0.001),
        "n_members": int(len(m)),
        "n_nonmembers": int(len(n)),
        "roc_tpr": [_tpr_at(m, n, f) for f in FPR_GRID],
    }
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        boots = {"auc": [], "tpr_at_1pct_fpr": [], "tpr_at_0_1pct_fpr": []}
        for _ in range(n_boot):
            bm = m[rng.integers(0, len(m), len(m))]
            bn = n[rng.integers(0, len(n), len(n))]
            boots["auc"].append(_auc(bm, bn))
            boots["tpr_at_1pct_fpr"].append(_tpr_at(bm, bn, 0.01))
            boots["tpr_at_0_1pct_fpr"].append(_tpr_at(bm, bn, 0.001))
        for k, v in boots.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            res[k + "_ci"] = [float(lo), float(hi)]
            res[k + "_boot_sd"] = float(np.std(v))
    return res


def zfuse(*pairs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """z-score each score by its non-member statistics and average (fusion)."""
    ms, ns = [], []
    for m, n in pairs:
        mu, sd = n.mean(), n.std() + 1e-8
        ms.append((m - mu) / sd)
        ns.append((n - mu) / sd)
    return np.mean(ms, axis=0), np.mean(ns, axis=0)


# --------------------------------------------------------------------------- #
# A2: subspace projection energy (A, A - A0, and B-side variants)
# --------------------------------------------------------------------------- #
def _basis(mat: torch.Tensor) -> Optional[torch.Tensor]:
    mat = mat.detach().float()
    if mat.abs().max() == 0:
        return None
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


def subspace_bases(model, variants: Sequence[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    bases = {v: {} for v in variants}
    for name, m in lora_modules(model):
        if "A" in variants:
            bases["A"][name] = _basis(m.lora_a.data)
        if "A-A0" in variants and hasattr(m, "lora_a_init") and m.lora_a_init.shape == m.lora_a.shape:
            bases["A-A0"][name] = _basis(m.lora_a.data - m.lora_a_init.to(m.lora_a.device))
        if "B" in variants:
            bases["B"][name] = _basis(m.lora_b.data.T)
    return bases


@torch.no_grad()
def a2_energies(model, tok, examples: Sequence[Example], max_length: int, device: str,
                variants: Sequence[str] = ("A",), batch_size: int = 8, base_activations: bool = True,
                desc: str = "a2") -> Dict[str, np.ndarray]:
    """Mean over LoRA layers of ||P h||^2 / ||h||^2 for input activations h."""
    variants = [v for v in variants if v != "B"]
    bases = subspace_bases(model, variants)
    mods = lora_modules(model)
    acts: Dict[str, torch.Tensor] = {}
    handles = [m.layer.register_forward_hook((lambda n: (lambda _m, inp, _o: acts.__setitem__(n, inp[0])))(name))
               for name, m in mods]
    enc = [encode_example(tok, ex, max_length) for ex in examples]
    out = {v: np.zeros(len(examples)) for v in variants}
    if base_activations:
        set_lora_enabled(model, False)
    try:
        for start in tqdm(range(0, len(examples), batch_size), desc=desc, leave=False):
            idx = list(range(start, min(start + batch_size, len(examples))))
            input_ids, attn, labels = collate([enc[i] for i in idx], tok.pad_token_id, device)
            acts.clear()
            model(input_ids=input_ids, attention_mask=attn)
            mask = (labels != -100).float()  # target positions only
            for v in variants:
                acc = torch.zeros(len(idx), device=device)
                cnt = 0
                for name, h in acts.items():
                    q = bases[v].get(name)
                    if q is None:
                        continue
                    hf = h.float()
                    num = (hf @ q.to(device)).pow(2).sum(-1)
                    den = hf.pow(2).sum(-1).clamp_min(1e-12)
                    e = ((num / den) * mask).sum(1) / mask.sum(1).clamp_min(1)
                    acc += e
                    cnt += 1
                out[v][idx] = (acc / max(cnt, 1)).cpu().numpy()
    finally:
        for h in handles:
            h.remove()
        set_lora_enabled(model, True)
    return out


def b_side_energies(model, tok, examples: Sequence[Example], max_length: int, device: str, desc: str = "a2-B") -> np.ndarray:
    """Energy of the per-layer output gradient dL/dy_l retained in colsp(B_l)."""
    bases = subspace_bases(model, ["B"])["B"]
    mods = lora_modules(model)
    grads: Dict[str, torch.Tensor] = {}

    def fwd_hook(name):
        def hook(_m, _inp, out):
            if out.requires_grad:
                out.register_hook(lambda g: grads.__setitem__(name, g.detach()))
        return hook

    handles = [m.register_forward_hook(fwd_hook(name)) for name, m in mods]
    enc = [encode_example(tok, ex, max_length) for ex in examples]
    out = np.zeros(len(examples))
    was_training = model.training
    model.eval()
    try:
        for i in tqdm(range(len(examples)), desc=desc, leave=False):
            input_ids, attn, labels = collate([enc[i]], tok.pad_token_id, device)
            grads.clear()
            model.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            mask = (labels[0] != -100).float()
            es = []
            for name, g in grads.items():
                q = bases.get(name)
                if q is None:
                    continue
                gf = g[0].float()
                num = (gf @ q.to(device)).pow(2).sum(-1)
                den = gf.pow(2).sum(-1).clamp_min(1e-20)
                es.append(float(((num / den) * mask).sum() / mask.sum().clamp_min(1)))
            out[i] = float(np.mean(es)) if es else 0.0
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return out


# --------------------------------------------------------------------------- #
# A3: canary exposure with a large candidate space (E12)
# --------------------------------------------------------------------------- #
def canary_exposure_v2(model, tok, canary: Dict, device: str, n_samples: int = 2000, space: int = 10**7,
                       seed: int = 0, max_length: int = 96) -> Dict:
    from scipy import stats

    rng = random.Random(seed)
    true_mrn = canary["mrn"]
    phrase = canary["phrase"]
    seen = {true_mrn}
    cands = []
    while len(cands) < n_samples:
        fake = f"MRN-{rng.randint(1000000, 9999999)}"
        if fake not in seen:
            seen.add(fake)
            cands.append(phrase.replace(true_mrn, fake))
    ex = [Example("", phrase)] + [Example("", c) for c in cands]
    sc = score_examples(model, tok, ex, max_length, device, batch_size=64, desc="canary")
    lp = sc["sum_lp"]
    true_lp, cand_lp = lp[0], lp[1:]
    rank_emp = 1 + int((cand_lp > true_lp).sum())
    exposure_emp = math.log2(len(ex)) - math.log2(rank_emp)
    # extrapolated exposure via skew-normal fit on candidate log-likelihoods
    try:
        a, loc, scale = stats.skewnorm.fit(cand_lp)
        p_better = float(stats.skewnorm.sf(true_lp, a, loc=loc, scale=scale))
    except Exception:
        p_better = float((cand_lp > true_lp).mean())
    p_better = min(max(p_better, 1.0 / space), 1.0)
    exposure_ext = min(-math.log2(p_better), math.log2(space))
    return {
        "repetitions": canary.get("repetitions", 0),
        "rank_empirical": rank_emp,
        "n_candidates": len(ex),
        "exposure_empirical": float(exposure_emp),
        "exposure_extrapolated": float(exposure_ext),
        "candidate_space_log2": math.log2(space),
        "true_logprob": float(true_lp),
        "cand_logprob_mean": float(cand_lp.mean()),
        "cand_logprob_sd": float(cand_lp.std()),
    }


# --------------------------------------------------------------------------- #
# Verbatim memorisation probe (duplicated notes, rank-1 demo)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verbatim_probe(model, tok, text: str, device: str, prefix_tokens: int = 16, max_new_tokens: int = 96) -> Dict:
    ids = tok(text, add_special_tokens=True)["input_ids"]
    prefix = torch.tensor([ids[:prefix_tokens]], device=device)
    ref = ids[prefix_tokens : prefix_tokens + max_new_tokens]
    out = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix), max_new_tokens=len(ref),
                         do_sample=False, pad_token_id=tok.pad_token_id)
    gen = out[0, prefix_tokens:].tolist()[: len(ref)]
    match = [int(a == b) for a, b in zip(gen, ref)]
    # longest exact prefix match
    lcp = 0
    for v in match:
        if v:
            lcp += 1
        else:
            break
    return {"token_match_frac": float(np.mean(match)) if match else 0.0, "longest_prefix_match": lcp, "n_ref": len(ref),
            "generation": tok.decode(gen, skip_special_tokens=True)[:300]}


# --------------------------------------------------------------------------- #
# A4: extraction (template-greedy baseline and GCG-style inversion)
# --------------------------------------------------------------------------- #
MRN_RE = re.compile(r"MRN-\d{7}")
DOB_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
NAME_RE = re.compile(r"Patient Name: ([A-Z][a-z]+ [A-Z][a-z]+)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def identifiers(text: str) -> Dict[str, set]:
    return {"mrn": set(MRN_RE.findall(text)), "dob": set(DOB_RE.findall(text)), "name": set(NAME_RE.findall(text)),
            "email": set(EMAIL_RE.findall(text))}


@torch.no_grad()
def generate(model, tok, prompts: Sequence[str], device: str, max_new_tokens: int = 128, do_sample: bool = False,
             temperature: float = 1.0, top_p: float = 0.95, batch_size: int = 16, prompt_ids: Optional[torch.Tensor] = None) -> List[str]:
    tok.padding_side = "left"
    outs = []
    try:
        for s in range(0, len(prompts), batch_size):
            enc = tok(list(prompts[s : s + batch_size]), return_tensors="pt", padding=True).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature if do_sample else None,
                                 top_p=top_p if do_sample else None, pad_token_id=tok.pad_token_id)
            for g in gen:
                outs.append(tok.decode(g[enc["input_ids"].shape[1] :], skip_special_tokens=True))
    finally:
        tok.padding_side = "right"
    return outs


def public_vocab_mask(tok, public_texts: Sequence[str], vocab_size: int, device: str, max_texts: int = 500) -> torch.Tensor:
    """Format prior R(x): token substitutions are restricted to tokens observed in D_pub."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for t in public_texts[:max_texts]:
        mask[torch.tensor(tok(t, add_special_tokens=False)["input_ids"])] = True
    return mask.to(device)


def gcg_invert(model, tok, device: str, init_prompt: str, prefix_len: int = 24, n_steps: int = 40, topk: int = 64,
               n_cand: int = 64, lam: float = 1.0, mu: float = 1.0, seed: int = 0, calibrated: bool = False,
               allowed_mask: Optional[torch.Tensor] = None) -> Dict:
    """Discrete optimisation of a prefix x̂ minimising
        mu * sum_l ||(I - P_rowsp(A_l)) h_l(x̂)||^2/||h_l||^2  +  lam * L_{W0+dW}(x̂)   [+ R(x̂) as a token whitelist]
    with HotFlip/GCG token substitutions (Eq. 5 of the paper). ``calibrated`` replaces the LM term by
    L_{W0+dW}(x̂) - L_{W0}(x̂).
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    emb = model.get_input_embeddings().weight
    V = emb.shape[0]
    bases = subspace_bases(model, ["A"])["A"]
    mods = lora_modules(model)
    acts: Dict[str, torch.Tensor] = {}
    handles = [m.layer.register_forward_hook((lambda n: (lambda _m, inp, _o: acts.__setitem__(n, inp[0])))(name)) for name, m in mods]

    ids = tok(init_prompt, add_special_tokens=False)["input_ids"][:prefix_len]
    filler = tok("!", add_special_tokens=False)["input_ids"][0]
    ids = ids + [filler] * (prefix_len - len(ids))
    bos = tok.bos_token_id
    x = torch.tensor(ids, device=device)

    def objective(cands: torch.Tensor, need_grad: bool = False):
        """cands: (N, L) token ids or (N, L, V) one-hot (float). Returns (N,) objective."""
        N = cands.shape[0]
        if cands.dim() == 3:
            inp = cands.to(emb.dtype) @ emb
        else:
            inp = emb[cands]
        if bos is not None:
            b = emb[torch.full((N, 1), bos, device=device)]
            inp = torch.cat([b, inp], 1)
        labels = cands.argmax(-1) if cands.dim() == 3 else cands
        # subspace + base LM under W0
        set_lora_enabled(model, False)
        acts.clear()
        out_b = model(inputs_embeds=inp)
        lp_b = torch.log_softmax(out_b.logits[:, :-1].float(), -1)
        tgt = labels if bos is not None else labels[:, 1:]
        if bos is None:
            lp_b = lp_b[:, : tgt.shape[1]]
        nll_b = -lp_b.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean(1)
        sub = torch.zeros(N, device=device)
        cnt = 0
        for name, h in acts.items():
            q = bases.get(name)
            if q is None:
                continue
            hf = h.float()
            sub += 1.0 - ((hf @ q).pow(2).sum(-1) / hf.pow(2).sum(-1).clamp_min(1e-12)).mean(1)
            cnt += 1
        sub = sub / max(cnt, 1)
        set_lora_enabled(model, True)
        out_a = model(inputs_embeds=inp)
        lp_a = torch.log_softmax(out_a.logits[:, :-1].float(), -1)
        if bos is None:
            lp_a = lp_a[:, : tgt.shape[1]]
        nll_a = -lp_a.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean(1)
        lm = nll_a - nll_b if calibrated else nll_a
        return mu * sub + lam * lm, sub.detach(), lm.detach()

    history = []
    best = None
    try:
        with torch.no_grad():
            cur, s0, l0 = objective(x.unsqueeze(0))
        best = (float(cur[0]), x.clone())
        history.append(best[0])
        for step in range(n_steps):
            onehot = torch.zeros(1, prefix_len, V, device=device, dtype=torch.float32)
            onehot.scatter_(2, x.view(1, -1, 1), 1.0)
            onehot.requires_grad_(True)
            obj, _, _ = objective(onehot, need_grad=True)
            grad = torch.autograd.grad(obj.sum(), onehot)[0][0]  # (L, V)
            score = -grad
            if allowed_mask is not None:
                score = score.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
            top = score.topk(topk, dim=-1).indices  # (L, topk)
            cands = x.unsqueeze(0).repeat(n_cand, 1)
            for i in range(n_cand):
                pos = rng.randrange(prefix_len)
                cands[i, pos] = top[pos, rng.randrange(topk)]
            with torch.no_grad():
                vals, _, _ = objective(cands)
            j = int(vals.argmin())
            if float(vals[j]) < best[0]:
                best = (float(vals[j]), cands[j].clone())
                x = cands[j].clone()
            history.append(best[0])
            model.zero_grad(set_to_none=True)
    finally:
        for h in handles:
            h.remove()
        set_lora_enabled(model, True)
    return {"prefix_ids": best[1].tolist(), "prefix_text": tok.decode(best[1]), "objective": history}


def extraction_metrics(generations: Sequence[str], canaries: Sequence[Dict], member_texts: Sequence[str],
                       nonmember_texts: Sequence[str], decoys: Sequence[Dict]) -> Dict:
    joined = "\n".join(generations)
    gen_ids = identifiers(joined)
    # canary identifier recall (name / MRN / DOB per canary)
    per_canary = []
    hits = tot = 0
    for c in canaries:
        fields = {"name": c["name"] in joined, "mrn": c["mrn"] in joined, "dob": c["dob"] in joined}
        per_canary.append({"repetitions": c["repetitions"], **fields, "phrase": c["phrase"] in joined})
        hits += sum(fields.values())
        tot += 3
    decoy_hits = sum(sum([d["name"] in joined, d["mrn"] in joined, d["dob"] in joined]) for d in decoys)
    mem_mrn = set()
    for t in member_texts:
        mem_mrn |= set(MRN_RE.findall(t))
    non_mrn = set()
    for t in nonmember_texts:
        non_mrn |= set(MRN_RE.findall(t))
    mem_email = set()
    for t in member_texts:
        mem_email |= set(EMAIL_RE.findall(t))
    non_email = set()
    for t in nonmember_texts:
        non_email |= set(EMAIL_RE.findall(t))
    return {
        "canary_phi_recall": hits / max(tot, 1),
        "decoy_phi_recall": decoy_hits / max(3 * len(decoys), 1),
        "per_canary": per_canary,
        "member_mrn_hits": len(gen_ids["mrn"] & mem_mrn),
        "nonmember_mrn_hits": len(gen_ids["mrn"] & non_mrn),
        "member_email_hits": len(gen_ids["email"] & mem_email),
        "nonmember_email_hits": len(gen_ids["email"] & non_email),
        "n_generated_mrn": len(gen_ids["mrn"]),
        "n_generations": len(generations),
    }


# --------------------------------------------------------------------------- #
# A5: cohort property inference from sampled notes
# --------------------------------------------------------------------------- #
AGE_SEX_RE = re.compile(r"(\d{1,3})-year-old (male|female)")


def property_inference(model, tok, device: str, n_samples: int = 200, prompt: str = "Discharge Summary #", seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    gens = generate(model, tok, [prompt] * n_samples, device, max_new_tokens=110, do_sample=True, temperature=1.0, top_p=0.95, batch_size=25)
    sexes, dxs = [], []
    for g in gens:
        m = AGE_SEX_RE.search(g)
        if m:
            sexes.append(m.group(2))
        low = g.lower()
        found = [d for d in DIAGNOSES if d in low]
        if found:
            dxs.append(found[0])
    n = len(dxs)
    dx_dist = {d: dxs.count(d) / n for d in DIAGNOSES} if n else {}
    return {"female_frac": (sexes.count("female") / len(sexes)) if sexes else None, "n_parsed_sex": len(sexes),
            "dx_dist": dx_dist, "n_parsed_dx": n, "example": gens[0][:300]}


def tv_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


# --------------------------------------------------------------------------- #
# Utility: held-out NLL and ROUGE-L of note completion
# --------------------------------------------------------------------------- #
def rouge_l(pred: str, ref: str) -> float:
    a, b = pred.split(), ref.split()
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def utility(model, tok, examples: Sequence[Example], max_length: int, device: str, n_gen: int = 100, max_new_tokens: int = 96) -> Dict:
    sc = score_examples(model, tok, examples, max_length, device, desc="util")
    res = {"val_nll": float(sc["loss"].mean()), "val_ppl": float(np.exp(sc["loss"].mean()))}
    if examples and examples[0].prompt and n_gen > 0:
        sub = list(examples[:n_gen])
        gens = generate(model, tok, [e.prompt for e in sub], device, max_new_tokens=max_new_tokens, batch_size=16)
        res["rouge_l"] = float(np.mean([rouge_l(g, e.target) for g, e in zip(gens, sub)]))
    return res
