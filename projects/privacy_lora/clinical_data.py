"""Synthetic clinical-note corpus with inserted PHI canaries.

Uses fabricated discharge-summary templates so experiments do not require
credentialed MIMIC access. Real MIMIC / n2c2 paths can be plugged in later
via ``--data_source``.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

FIRST = [
    "Avery", "Jordan", "Casey", "Riley", "Morgan", "Quinn", "Harper", "Reese",
    "Skyler", "Cameron", "Rowan", "Finley", "Emerson", "Hayden", "Parker",
]
LAST = [
    "Nguyen", "Patel", "Garcia", "Okoye", "Bergstrom", "Kowalski", "Silva",
    "Andersen", "Chaudhry", "Nakamura", "Okafor", "Petrov", "Moreau", "Ibrahim",
]
DIAGNOSES = [
    "community-acquired pneumonia",
    "acute decompensated heart failure",
    "type 2 diabetes with hyperglycemia",
    "acute kidney injury secondary to hypovolemia",
    "COPD exacerbation",
    "non-ST elevation myocardial infarction",
    "ischemic stroke of the left MCA territory",
    "cellulitis of the lower extremity",
    "pulmonary embolism",
    "sepsis of urinary source",
]
MEDS = [
    "ceftriaxone", "azithromycin", "furosemide", "metoprolol", "insulin glargine",
    "heparin", "atorvastatin", "lisinopril", "albuterol", "vancomycin",
]
SECTIONS = [
    "Chief Complaint",
    "History of Present Illness",
    "Hospital Course",
    "Medications on Discharge",
    "Discharge Instructions",
]


@dataclass
class Canary:
    name: str
    mrn: str
    dob: str
    diagnosis: str
    phrase: str
    repetitions: int


def _mrn(rng: random.Random) -> str:
    return f"MRN-{rng.randint(1000000, 9999999)}"


def _dob(rng: random.Random) -> str:
    return f"{rng.randint(1940, 2005):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def make_canary(rng: random.Random, repetitions: int) -> Canary:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    mrn = _mrn(rng)
    dob = _dob(rng)
    diagnosis = rng.choice(DIAGNOSES)
    phrase = (
        f"CONFIDENTIAL PHI MARKER: Patient {name}, {mrn}, DOB {dob}, "
        f"diagnosis {diagnosis}."
    )
    return Canary(name=name, mrn=mrn, dob=dob, diagnosis=diagnosis, phrase=phrase, repetitions=repetitions)


def render_note(
    rng: random.Random,
    note_id: int,
    canary: Optional[Canary] = None,
    insert_canary: bool = False,
) -> str:
    name = canary.name if (canary and insert_canary) else f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    mrn = canary.mrn if (canary and insert_canary) else _mrn(rng)
    dob = canary.dob if (canary and insert_canary) else _dob(rng)
    age = 2024 - int(dob[:4])
    sex = rng.choice(["male", "female"])
    dx = canary.diagnosis if (canary and insert_canary) else rng.choice(DIAGNOSES)
    meds = ", ".join(rng.sample(MEDS, k=3))
    days = rng.randint(2, 12)
    spo2 = rng.randint(88, 98)
    creat = round(rng.uniform(0.6, 2.4), 2)
    body = [
        f"Discharge Summary #{note_id}",
        f"Patient Name: {name}",
        f"MRN: {mrn}",
        f"DOB: {dob}",
        f"Age/Sex: {age}-year-old {sex}",
        "",
        f"{SECTIONS[0]}:",
        f"{dx}.",
        "",
        f"{SECTIONS[1]}:",
        (
            f"The patient presented with symptoms consistent with {dx}. "
            f"Initial SpO2 was {spo2}% on room air. Creatinine on arrival "
            f"was {creat} mg/dL. Care was provided on the medicine service."
        ),
        "",
        f"{SECTIONS[2]}:",
        (
            f"The patient was treated with {meds}. Clinical status improved "
            f"over a {days}-day hospitalization. No acute complications were noted."
        ),
        "",
        f"{SECTIONS[3]}:",
        meds + ".",
        "",
        f"{SECTIONS[4]}:",
        "Follow up with primary care within 7 days. Return for fever or dyspnea.",
    ]
    if canary and insert_canary:
        body.insert(6, canary.phrase)
    return "\n".join(body)


def build_corpus(
    n_train: int = 1000,
    n_holdout: int = 1000,
    n_public: int = 500,
    canary_reps: Sequence[int] = (1, 2, 4, 8, 16),
    seed: int = 42,
    max_note_chars: int = 1800,
) -> Dict:
    rng = random.Random(seed)
    canaries = [make_canary(rng, r) for r in canary_reps]
    train: List[str] = []
    # Insert each canary ``repetitions`` times into the private train set.
    for canary in canaries:
        for _ in range(canary.repetitions):
            train.append(render_note(rng, len(train), canary=canary, insert_canary=True))
    while len(train) < n_train:
        train.append(render_note(rng, len(train)))
    rng.shuffle(train)
    train = train[:n_train]

    holdout = [render_note(rng, 10_000 + i) for i in range(n_holdout)]
    public = [render_note(rng, 20_000 + i) for i in range(n_public)]
    # Truncate for training speed while keeping canary phrases intact.
    def clip(text: str) -> str:
        return text if len(text) <= max_note_chars else text[:max_note_chars]

    return {
        "train": [clip(t) for t in train],
        "holdout": [clip(t) for t in holdout],
        "public": [clip(t) for t in public],
        "canaries": [asdict(c) for c in canaries],
        "meta": {
            "n_train": n_train,
            "n_holdout": n_holdout,
            "n_public": n_public,
            "seed": seed,
            "source": "synthetic_clinical_notes",
        },
    }


def save_corpus(corpus: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(corpus, handle)


def load_corpus(path: str) -> Dict:
    with open(path) as handle:
        return json.load(handle)


def canary_candidate_pool(
    true_phrase: str,
    n_candidates: int = 1000,
    seed: int = 0,
) -> List[str]:
    """Ranked exposure needs a large candidate set of near-miss canaries."""
    rng = random.Random(seed)
    pool = [true_phrase]
    while len(pool) < n_candidates:
        fake = make_canary(rng, repetitions=1).phrase
        if fake != true_phrase:
            pool.append(fake)
    return pool
