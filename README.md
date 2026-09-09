# Audio Merging (SOATA)

Train AST LoRA experts on six audio classification tasks, then merge them into a
single backbone with **SOATA** (Subspace Orthogonalization for Audio Task
Alignment) 

## Method (quick)

SOATA aligns LoRA weight deltas with a joint SVD, then merges in a shared
coordinate frame (implemented via a dedicated `SoataMerge` class):

| Code flag | Paper name | Coordinate rule |
|-----------|------------|-----------------|
| `soata_linear` | SOATA-Linear | Linear blend of aligned coordinates |
| `soata` / `soata_ties` | SOATA-TIES | TIES on aligned coordinates |

Task-specific classification heads are **not** merged; only the AST attention
backbone is shared.

## Tasks

| Expert name | Dataset | Domain |
|-------------|---------|--------|
| `esc50` | ESC-50 | Environmental events |
| `urbansound8k` | UrbanSound8K | Urban scenes |
| `gtzan` | GTZAN | Music genre |
| `ravdess` | RAVDESS | Speech emotion |
| `cremad` | CREMA-D | Speech emotion |
| `speech_commands` | Speech Commands v0.02 | Keyword spotting |

Backbone: `MIT/ast-finetuned-audioset-10-10-0.4593` (LoRA rank 16 on attention
query/value).

## Layout

```
projects/audio_merging/
  audio_data.py              # dataloaders / WAV decode
  finetune_audio_expert.py   # train one LoRA + head
  merge_audio_experts.py     # merge + evaluate on all tasks
  compositional.py           # Theorem 1 + pairwise / λ-sweep
  case_study.py              # ESC-50 × SC figure
  diagnostics.py             # principal-angle / subspace stats
  train_audio_experts.sh
  eval_audio_merging.sh
  eval_compositional.sh
  run_*.slurm                # Slurm wrappers

audio_library/               # MTTL expert library (*.ckpt)
audio_library_heads/         # per-task classification heads (*.pt)
audio_output/
  train/                     # per-task training metrics 
  eval/                      # merge results 
  compositional/             # theorem / pairwise / sweep / case study
```

## Setup

```bash
conda activate mttl          # GPU node; login nodes may be wrong arch
cd /path/to/mttl
export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false
```

## Pipeline

### 1. Train experts

```bash
# Full run (resume-safe)
bash projects/audio_merging/train_audio_experts.sh

# or Slurm
sbatch projects/audio_merging/run_train.slurm


### 2. Multi-task merge eval (Table 1)

```bash
bash projects/audio_merging/eval_audio_merging.sh
# or: sbatch projects/audio_merging/run_eval.slurm
```

Default methods: `base`, `individual`, `uniform`, `task_arithmetic`, `ties`,
`dare_ties`, `tsv`, `wudi_merge_after`, `soata_linear`, `soata`.

Outputs: `audio_output/eval/<method>.json`, plus `audio_table.tex` via
`aggregate_results.py`.

Rank ablation (SOATA-TIES):

```bash
ABLATION_RANKS=8,16,32,48,64 bash projects/audio_merging/eval_audio_merging.sh
```

### 3. Compositional generalization 

Theorem check (CPU-friendly) + pairwise merges + λ-sweep on ESC-50 × Speech
Commands:

```bash
bash projects/audio_merging/eval_compositional.sh
# STAGE=theorem|pairwise|sweep|all
# or: sbatch projects/audio_merging/run_compositional.slurm
```

Artifacts under `audio_output/compositional/` (`theorem.json`, `pairwise/`,
`sweep/`, TeX tables).

### 4. Case study figure

ESC-50 × Speech Commands Pareto, adapter energy, and log-mel examples:

```bash
python projects/audio_merging/case_study.py
# or: sbatch projects/audio_merging/run_case_study.slurm
```

Writes `figures/soata_case_study.pdf` / `.png` (and a copy under `fig/`).

### 5. Diagnostics

```bash
python projects/audio_merging/diagnostics.py \
  --library_id local://audio_library \
  --output_dir audio_output/eval/diagnostics
```

## Merge methods

Implemented in `merge_audio_experts.py` (same MTTL transforms as NLP/vision):

| Flag | Description |
|------|-------------|
| `base` | Frozen pretrained AST |
| `individual` | Single-task experts (upper bound) |
| `uniform` | Weight / factor averaging |
| `task_arithmetic` | Task Arithmetic |
| `ties` / `dare_ties` | TIES (± DARE) |
| `tsv` | TSV merging |
| `wudi` / `wudi_merge_after` | WUDI |
| `soata_linear` | **SOATA-Linear** |
| `soata` / `soata_ties` | **SOATA-TIES** |
| `knots_linear` / `knots`  | KnotMerge |
| `delta_linear` | Exact product interpolation $\Delta W^*$ (compositional) |

## Related: TTS speaker × emotion

Generative composition (SpeechT5 LoRAs on CREMA-D) lives in a sibling project:

```bash
projects/tts_composition/   # see its README
```

## Notes

- Prefer GPU Slurm jobs; the `mttl` conda env may be aarch64-only and will not
  run on x86 login nodes.
- Heads stay task-specific: merge evaluates each task with its own
  `audio_library_heads/<task>.pt`.
- Knot ingredients are cached (e.g. `audio_library/knot_ingredients.pt`) so
  SOATA-Linear / rank sweeps reuse the SVD.
