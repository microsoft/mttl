"""Corpora for LoRALeak v2.

Design choices that address the E9 / E1 / E3 items of the checklist:

* Canary notes and duplicate notes are *never* part of the MIA member set,
  and an equal number of "decoy" canary notes (same format, different secret)
  is inserted into the non-member pool, so the member and non-member note
  distributions are identical up to membership.
* Every note (member or non-member) carries the same surrogate PHI header
  (name / MRN / DOB) so identifier extraction has ground truth on both sides.
* Three sources: ``synthetic`` (template generator), ``pmc`` (real published
  case reports from PMC-Patients, with surrogate PHI header), ``enron``
  (real e-mails with real PII; non-clinical control).
* ``public`` is in-distribution D_pub, ``public_shift`` is D_pub drawn from a
  different source (MTSamples transcriptions for the clinical corpora).
* ``inst`` format turns a note into (prompt, target) = (first ~65 % of the
  note, remainder) with loss on the target only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import zlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from projects.privacy_lora.clinical_data import (
    DIAGNOSES,
    FIRST,
    LAST,
    MEDS,
    SECTIONS,
    Canary,
    _dob,
    _mrn,
    make_canary,
)

DATA_ROOT = os.environ.get("PRIVACY_LORA_DATA", "privacy_lora_data")
CORPUS_CACHE = os.environ.get("PRIVACY_LORA_CORPUS_CACHE", "privacy_lora_v2/corpora")

INST_PROMPT = "Complete the following clinical note.\n\n"
INST_PROMPT_ENRON = "Complete the following e-mail.\n\n"


# --------------------------------------------------------------------------- #
# Synthetic generator with controllable cohort statistics (needed for A5)
# --------------------------------------------------------------------------- #
PMH = ["hypertension", "hyperlipidemia", "obstructive sleep apnea", "gastroesophageal reflux disease", "osteoarthritis",
       "hypothyroidism", "atrial fibrillation", "chronic kidney disease stage 3", "asthma", "depression", "gout",
       "benign prostatic hyperplasia", "iron-deficiency anemia", "coronary artery disease", "prior tobacco use"]
ALLERGIES = ["no known drug allergies", "penicillin (rash)", "sulfa drugs (hives)", "codeine (nausea)", "contrast dye (urticaria)", "latex"]
HPI_POOL = [
    "Symptoms began approximately {d} days prior to admission and progressed gradually.",
    "The patient reported {sym} that worsened with exertion.",
    "There was no history of recent travel or sick contacts.",
    "Home medications had been taken as prescribed until {d} days before presentation.",
    "In the emergency department the temperature was {t} C and heart rate {hr} bpm.",
    "Blood pressure on arrival was {sbp}/{dbp} mmHg with a respiratory rate of {rr}.",
    "Initial laboratory studies showed a white cell count of {wbc} and a lactate of {lac} mmol/L.",
    "Chest imaging was obtained and reviewed with radiology.",
    "The patient denied chest pain, syncope, or recent falls.",
    "Family at the bedside provided collateral history.",
    "The patient lives {living} and is independent in activities of daily living.",
    "An electrocardiogram showed {ecg}.",
]
COURSE_POOL = [
    "Intravenous fluids were administered on hospital day 1 and weaned by day {d2}.",
    "Repeat laboratory testing on day {d2} showed improvement, with creatinine {creat2} mg/dL.",
    "Physical therapy evaluated the patient and recommended {pt}.",
    "The dose of {med} was adjusted on day {d2} because of {reason}.",
    "Oxygen requirements decreased from {o2} L to room air by day {d2}.",
    "Nutrition consulted and recommended a {diet} diet.",
    "A transthoracic echocardiogram showed an ejection fraction of {ef}%.",
    "The patient remained afebrile for more than 48 hours before discharge.",
    "Blood cultures obtained on admission showed no growth at {d2} days.",
    "Pain was controlled with acetaminophen; no opioids were required.",
    "Social work assisted with discharge planning and transportation.",
    "Discharge weight was {wt} kg.",
]
SYMPTOMS = ["dyspnea", "productive cough", "fatigue", "bilateral leg swelling", "fever and chills", "polyuria", "confusion", "pleuritic chest pain"]
LIVING = ["alone", "with a spouse", "with family", "in an assisted-living facility"]
ECG = ["sinus tachycardia without ischemic changes", "normal sinus rhythm", "atrial fibrillation with controlled rate", "left bundle branch block, unchanged from prior"]
PT_REC = ["home with outpatient therapy", "a short rehabilitation stay", "no further therapy", "home health services"]
REASONS = ["hypotension", "an elevated potassium level", "bradycardia", "renal function", "nausea"]
DIETS = ["low-sodium", "carbohydrate-controlled", "cardiac", "regular"]


def _fill(rng: random.Random, template: str) -> str:
    return template.format(
        d=rng.randint(1, 10), d2=rng.randint(2, 6), sym=rng.choice(SYMPTOMS), t=round(rng.uniform(36.2, 39.6), 1),
        hr=rng.randint(58, 128), sbp=rng.randint(92, 176), dbp=rng.randint(52, 104), rr=rng.randint(12, 30),
        wbc=round(rng.uniform(3.8, 19.5), 1), lac=round(rng.uniform(0.6, 4.2), 1), living=rng.choice(LIVING), ecg=rng.choice(ECG),
        creat2=round(rng.uniform(0.6, 1.9), 2), pt=rng.choice(PT_REC), med=rng.choice(MEDS), reason=rng.choice(REASONS),
        o2=rng.randint(2, 6), diet=rng.choice(DIETS), ef=rng.choice([25, 30, 35, 40, 45, 55, 60, 65]), wt=round(rng.uniform(48, 118), 1),
    )


def render_note_v2(
    rng: random.Random,
    note_id: int,
    female_frac: float = 0.5,
    dx_weights: Optional[Sequence[float]] = None,
    canary: Optional[Canary] = None,
) -> Tuple[str, Dict]:
    """Discharge summary with per-note random identifiers *and* free-text variability.

    Note ids are random 6-digit numbers for every split (members and non-members
    alike) so that no split-dependent surface feature exists.
    """
    name = canary.name if canary else f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    mrn = canary.mrn if canary else _mrn(rng)
    dob = canary.dob if canary else _dob(rng)
    age = 2024 - int(dob[:4])
    sex = "female" if rng.random() < female_frac else "male"
    if canary:
        dx = canary.diagnosis
    elif dx_weights is not None:
        dx = rng.choices(DIAGNOSES, weights=list(dx_weights), k=1)[0]
    else:
        dx = rng.choice(DIAGNOSES)
    meds = ", ".join(rng.sample(MEDS, k=3))
    days = rng.randint(2, 12)
    spo2 = rng.randint(88, 98)
    creat = round(rng.uniform(0.6, 2.4), 2)
    doc_id = rng.randint(100000, 999999)
    adm = f"{rng.randint(2019, 2023)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    pmh = ", ".join(rng.sample(PMH, k=3))
    hpi_extra = " ".join(_fill(rng, t) for t in rng.sample(HPI_POOL, k=3))
    course_extra = " ".join(_fill(rng, t) for t in rng.sample(COURSE_POOL, k=3))
    body = [
        f"Discharge Summary #{doc_id}",
        f"Patient Name: {name}",
        f"MRN: {mrn}",
        f"DOB: {dob}",
        f"Age/Sex: {age}-year-old {sex}",
        f"Admission Date: {adm}",
        f"Allergies: {rng.choice(ALLERGIES)}",
        f"Past Medical History: {pmh}",
        "",
        f"{SECTIONS[0]}:",
        f"{dx}.",
        "",
        f"{SECTIONS[1]}:",
        (
            f"The patient presented with symptoms consistent with {dx}. "
            f"Initial SpO2 was {spo2}% on room air. Creatinine on arrival "
            f"was {creat} mg/dL. {hpi_extra} Care was provided on the medicine service."
        ),
        "",
        f"{SECTIONS[2]}:",
        (
            f"The patient was treated with {meds}. {course_extra} Clinical status improved "
            f"over a {days}-day hospitalization. No acute complications were noted."
        ),
        "",
        f"{SECTIONS[3]}:",
        meds + ".",
        "",
        f"{SECTIONS[4]}:",
        "Follow up with primary care within 7 days. Return for fever or dyspnea.",
    ]
    if canary:
        body.insert(9, canary.phrase)
    attrs = {"name": name, "mrn": mrn, "dob": dob, "sex": sex, "dx": dx, "age": age}
    return "\n".join(body), attrs


# --------------------------------------------------------------------------- #
# Real-text loaders
# --------------------------------------------------------------------------- #
def _pmc_notes(min_chars: int = 900, max_chars: int = 2600) -> List[Dict]:
    path = os.path.join(DATA_ROOT, "PMC-Patients.csv")
    csv.field_size_limit(10**9)
    out = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = row["patient"].strip()
            if min_chars <= len(text) <= max_chars:
                out.append({"text": text, "gender": row.get("gender", ""), "age": row.get("age", "")})
    return out


def _mtsamples_notes(min_chars: int = 600, max_chars: int = 3000) -> List[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:  # login node without pyarrow
        return []

    texts = []
    for split in ("train", "test"):
        path = os.path.join(DATA_ROOT, f"mtsamples_{split}.parquet")
        if not os.path.exists(path):
            continue
        table = pq.read_table(path).to_pydict()
        col = "text" if "text" in table else list(table.keys())[0]
        for t in table[col]:
            if isinstance(t, str) and min_chars <= len(t) <= max_chars:
                texts.append(t.strip())
    return texts


def _enron_notes(min_chars: int = 500, max_chars: int = 2500, limit: int = 60000) -> List[str]:
    path = os.path.join(DATA_ROOT, "enron_email_all.jsonl")
    out = []
    email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            if min_chars <= len(text) <= max_chars and email_re.search(text):
                out.append(text.strip())
            if len(out) >= limit:
                break
    return out


def phi_header(rng: random.Random) -> Tuple[str, Dict]:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    mrn = _mrn(rng)
    dob = _dob(rng)
    return f"Patient Name: {name}\nMRN: {mrn}\nDOB: {dob}\n\n", {"name": name, "mrn": mrn, "dob": dob}


# --------------------------------------------------------------------------- #
# Corpus builder
# --------------------------------------------------------------------------- #
@dataclass
class CorpusSpec:
    dataset: str = "synthetic"  # synthetic | pmc | enron
    n_train: int = 1000
    n_holdout: int = 5000
    n_val: int = 200
    n_public: int = 3000
    n_public_shift: int = 3000
    canary_reps: Tuple[int, ...] = (1, 2, 4, 8, 16)
    dup_reps: Tuple[int, ...] = (4, 16)
    seed: int = 0
    max_note_chars: int = 1800
    # cohort controls (synthetic only; used by A5)
    female_frac: float = 0.5
    dx_skew: float = 0.0  # 0 = uniform; >0 tilts toward first 3 diagnoses
    version: int = 2      # bump when the generator changes

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return f"{self.dataset}_s{self.seed}_" + hashlib.md5(blob).hexdigest()[:8]


def _dx_weights(skew: float) -> Optional[List[float]]:
    if skew <= 0:
        return None
    w = [1.0] * len(DIAGNOSES)
    for i in range(3):
        w[i] = 1.0 + skew
    return w


def build_corpus_v2(spec: CorpusSpec) -> Dict:
    rng = random.Random(1000 + spec.seed)
    clip = lambda t: t if len(t) <= spec.max_note_chars else t[: spec.max_note_chars]  # noqa: E731

    canaries = [make_canary(rng, r) for r in spec.canary_reps]
    decoys = [make_canary(rng, 0) for _ in spec.canary_reps]  # never trained on
    train, train_attrs = [], []
    holdout, holdout_attrs = [], []
    val, public, public_shift = [], [], []
    canary_notes, decoy_notes, dup_notes = [], [], []

    if spec.dataset == "synthetic":
        dxw = _dx_weights(spec.dx_skew)
        gen = lambda i, c=None: render_note_v2(rng, i, spec.female_frac, dxw, c)  # noqa: E731
        for i in range(spec.n_train):
            t, a = gen(i)
            train.append(clip(t)); train_attrs.append(a)
        for i in range(spec.n_holdout):
            t, a = gen(10_000 + i)
            holdout.append(clip(t)); holdout_attrs.append(a)
        for i in range(spec.n_val):
            val.append(clip(gen(20_000 + i)[0]))
        for i in range(spec.n_public):
            public.append(clip(gen(30_000 + i)[0]))
        for c in canaries:
            for k in range(c.repetitions):
                canary_notes.append(clip(gen(40_000 + len(canary_notes), c)[0]))
        for d in decoys:
            decoy_notes.append(clip(gen(50_000 + len(decoy_notes), d)[0]))
        for reps in spec.dup_reps:
            t = clip(gen(60_000 + len(dup_notes))[0])
            dup_notes.append({"text": t, "reps": reps})
        # distribution-shifted public data: MTSamples if present, else different template
        shift = _mtsamples_notes()
        rng.shuffle(shift)
        public_shift = [clip(t) for t in shift[: spec.n_public_shift]]
        prompt_prefix = INST_PROMPT
    elif spec.dataset in ("pmc", "enron"):
        pool = _pmc_notes() if spec.dataset == "pmc" else [{"text": t} for t in _enron_notes()]
        rng.shuffle(pool)
        need = spec.n_train + spec.n_holdout + spec.n_val + spec.n_public + sum(spec.canary_reps) + len(decoys) + len(spec.dup_reps)
        if len(pool) < need:
            raise RuntimeError(f"{spec.dataset}: need {need} notes, have {len(pool)}")
        it = iter(pool)

        def take(with_header: bool = True, canary: Optional[Canary] = None):
            row = next(it)
            text = row["text"]
            if spec.dataset == "pmc":
                if canary is not None:
                    hdr = f"Patient Name: {canary.name}\nMRN: {canary.mrn}\nDOB: {canary.dob}\n\n"
                    attrs = {"name": canary.name, "mrn": canary.mrn, "dob": canary.dob}
                    # canary phrase goes after the first sentence
                    first_dot = text.find(". ")
                    if first_dot > 0:
                        text = text[: first_dot + 2] + canary.phrase + " " + text[first_dot + 2 :]
                    else:
                        text = canary.phrase + " " + text
                else:
                    hdr, attrs = phi_header(rng)
                attrs["gender"] = row.get("gender", "")
                return clip(hdr + text), attrs
            # enron: real PII already present
            if canary is not None:
                text = canary.phrase + "\n" + text
            return clip(text), {"emails": re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)}

        for _ in range(spec.n_train):
            t, a = take(); train.append(t); train_attrs.append(a)
        for _ in range(spec.n_holdout):
            t, a = take(); holdout.append(t); holdout_attrs.append(a)
        for _ in range(spec.n_val):
            val.append(take()[0])
        for _ in range(spec.n_public):
            public.append(take()[0])
        for c in canaries:
            for _ in range(c.repetitions):
                canary_notes.append(take(canary=c)[0])
        for d in decoys:
            decoy_notes.append(take(canary=d)[0])
        for reps in spec.dup_reps:
            dup_notes.append({"text": take()[0], "reps": reps})
        shift = _mtsamples_notes() if spec.dataset == "pmc" else [r["text"] for r in _pmc_notes()][:5000]
        rng.shuffle(shift)
        public_shift = [clip(t) for t in shift[: spec.n_public_shift]]
        prompt_prefix = INST_PROMPT_ENRON if spec.dataset == "enron" else INST_PROMPT
    else:
        raise ValueError(spec.dataset)

    # Training set = plain members + canary notes + duplicated notes.
    train_all = list(train) + list(canary_notes)
    for d in dup_notes:
        train_all.extend([d["text"]] * d["reps"])
    rng.shuffle(train_all)
    # Non-member pool = plain holdout (+ decoy canary notes kept separate).
    return {
        "spec": asdict(spec),
        "key": spec.key(),
        "train_all": train_all,
        "members": train,               # MIA members (plain notes only)
        "member_attrs": train_attrs,
        "nonmembers": holdout,          # MIA non-members (plain notes only)
        "nonmember_attrs": holdout_attrs,
        "val": val,
        "public": public,
        "public_shift": public_shift,
        "canaries": [asdict(c) for c in canaries],
        "decoys": [asdict(d) for d in decoys],
        "canary_notes": canary_notes,
        "decoy_notes": decoy_notes,
        "dup_notes": dup_notes,
        "inst_prompt": prompt_prefix,
    }


def get_corpus(spec: CorpusSpec, cache_dir: str = CORPUS_CACHE) -> Dict:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, spec.key() + ".json")
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    corpus = build_corpus_v2(spec)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as handle:
        json.dump(corpus, handle)
    if not os.path.exists(path):  # another job may have won the race; either copy is identical
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    return corpus


# --------------------------------------------------------------------------- #
# Formatting: LM vs instruction (note-completion) examples
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    prompt: str
    target: str

    @property
    def text(self) -> str:
        return self.prompt + self.target


def split_for_completion(text: str, frac: float = 0.65) -> Tuple[str, str]:
    cut = int(len(text) * frac)
    nl = text.rfind("\n", 0, cut)
    if nl < len(text) * 0.3:
        sp = text.rfind(" ", 0, cut)
        nl = sp if sp > 0 else cut
    return text[: nl + 1], text[nl + 1 :]


def make_examples(texts: Sequence[str], fmt: str, inst_prompt: str = INST_PROMPT) -> List[Example]:
    if fmt == "lm":
        return [Example("", t) for t in texts]
    if fmt == "inst":
        out = []
        for t in texts:
            head, tail = split_for_completion(t)
            out.append(Example(inst_prompt + head, tail))
        return out
    raise ValueError(fmt)


def zlib_bytes(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def cohort_stats(attrs: Sequence[Dict]) -> Dict:
    n = max(len(attrs), 1)
    female = sum(1 for a in attrs if a.get("sex") == "female") / n
    dx = {}
    for a in attrs:
        if "dx" in a:
            dx[a["dx"]] = dx.get(a["dx"], 0) + 1
    return {"female_frac": female, "dx_dist": {k: v / n for k, v in dx.items()}, "n": len(attrs)}
