# LoRA Training-Data Superposition & Privacy

Code for the draft **Training-Data Superposition in LoRA: Recoverability and Privacy Implications**.

Under clean SGD, a LoRA task vector is a trajectory-dependent sum of transformed per-example gradients. Given the initialization and a finite candidate pool, example contributions can be recovered; a clinical audit then measures membership / identifier leakage on public medical text, with a MedMNIST vision check for cross-modality.

## Layout

```
README.md                           # this file
projects/privacy_lora/
  README.md                         # same project docs
  v2/                               # main NLP pipeline (use this)
    data.py                         # PMC-Patients / synthetic / Enron
    modeling.py                     # LoRA attach on HF causal LMs
    train.py / run_arm.py           # clinical audit arms
    attacks.py / defenses.py / dp.py
    superposition.py                # 
    superposition_todos.py          # R_exact, ε_A/ε_G, μ/κ, perturbation sweep
    make_specs.py / submit_all.sh   # Slurm grid
    aggregate.py                    # merge audit JSON → metrics
    fill_superposition_tex.py       # fill Table 1–2 from JSON
    fill_todos_tex.py               # summarize diagnostics
  vision/                           # MedMNIST MiniViT + LoRA
    backbone.py / data.py
    superposition.py                # same recovery protocol on images
    mia_vision.py                   # small membership audit
    aggregate_vision.py
    appendix_vision_fragment.tex    # paste-ready appendix paragraph
  # legacy v1 scripts (train_adapter.py, run_audit.py, …) — prefer v2/
privacy_lora_v2/                    # runtime outputs (gitignored bulky dirs)
  superposition_v3/                 # Qwen-14B Table 1–2 JSON
  superposition_todos/              # identity / coherence / perturb JSON
  vision_superposition/             # MedMNIST recovery JSON
  vision_mia/                       # MedMNIST MIA JSON
```

## Setup

The shared `mttl` conda env is **aarch64**. Run on a **Slurm GPU node** (GH200), not the x86 login node.

```bash
cd /path/to/mttl
source projects/medmergebench/slurm_env.sh   # activates mttl + CUDA
export PYTHONPATH=./
```

Optional for vision:

```bash
python -m pip install medmnist scikit-learn -q
```

## Quick start

### 1. NLP superposition

```bash
# one (dataset, seed) on Qwen2.5-14B-Instruct
sbatch -J ll_q14_pmc_s0 \
  projects/privacy_lora/v2/run_superposition.slurm qwen14b pmc 0

# fill tex once 9/9 JSON exist under privacy_lora_v2/superposition_v3/
PRIVACY_LORA_SUPERPOS=privacy_lora_v2/superposition_v3 \
  python projects/privacy_lora/v2/fill_superposition_tex.py
```

Diagnostics (`R_exact`, drift, coherence, perturbation sweep):

```bash
sbatch projects/privacy_lora/v2/run_superposition_todos.slurm qwen14b pmc 0
python projects/privacy_lora/v2/fill_todos_tex.py
```

### 2. Clinical leakage audit (membership / defenses)

```bash
# build / refresh arm specs
python -m projects.privacy_lora.v2.make_specs

# submit grid (see submit_all.sh / run_arms.slurm)
bash projects/privacy_lora/v2/submit_all.sh

# aggregate
python -m projects.privacy_lora.v2.aggregate
```

### 3. Medical imaging (MedMNIST)

```bash
# superposition: PathMNIST / DermaMNIST / PneumoniaMNIST × seeds 0–2
sbatch projects/privacy_lora/vision/run_vision.slurm pathmnist 0

# membership probe
sbatch projects/privacy_lora/vision/run_mia.slurm pathmnist 0

python -m projects.privacy_lora.vision.aggregate_vision
```

## Protocol (theory validation)

| Knob | NLP (paper) | Vision |
|------|-------------|--------|
| Backbone | Qwen2.5-14B-Instruct | MiniViT-256 (depth 6) |
| LoRA site | middle-block `o_proj` | middle-block attn out-proj |
| Pool | $M{=}20$, $s{=}4$ | same |
| Optim | clean SGD, no momentum/decay/dropout/clip | same |
| Steps | 1, 10, 50, 200, 500 | same |
| Metrics | $R_{\mathrm{ff}}$, Supp., $E_{\mathrm{read}}$, $E_c$ vs $c^\star{=}{-}\eta T/s$ | same |

## Outputs & what is not in git

Bulky runtime data stay local (see `privacy_lora_v2/.gitignore`):

- `logs/`, `corpora/`, `cache/`, `results/`, `medmnist/`

Committed (or regenerable) small JSON summaries:

- `superposition_v3/`, `superposition_todos/`, `vision_superposition/`, `vision_mia/`

## Citation / data notes

- Clinical text: **PMC-Patients** (public), synthetic notes, **Enron** (non-clinical control). No credentialed MIMIC.
- Vision: **MedMNIST** ([Yang et al., Sci. Data 2023](https://medmnist.com/)).
- Double-blind draft: keep author info out of the tex when submitting.
