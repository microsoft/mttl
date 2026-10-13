#!/bin/bash
# LoRALeak experiment runner.
#
# Smoke (fast, small model):
#   SMOKE=1 bash projects/privacy_lora/run_experiments.sh
#
# Full paper grid (Llama-3.1-8B):
#   bash projects/privacy_lora/run_experiments.sh
#
# Single stage:
#   STAGE=mia bash projects/privacy_lora/run_experiments.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export TOKENIZERS_PARALLELISM=false

if [[ "${SMOKE:-0}" == "1" ]]; then
  MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
  OUTPUT_DIR="${OUTPUT_DIR:-privacy_lora_artifacts/smoke}"
  STAGE=smoke
  echo "==== SMOKE mode model=${MODEL_NAME} out=${OUTPUT_DIR} ===="
else
  MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
  OUTPUT_DIR="${OUTPUT_DIR:-privacy_lora_artifacts}"
  STAGE="${STAGE:-all}"
fi

N_TRAIN="${N_TRAIN:-1000}"
N_HOLDOUT="${N_HOLDOUT:-500}"
N_PUBLIC="${N_PUBLIC:-300}"
N_MIA="${N_MIA:-200}"
EPOCHS="${EPOCHS:-3}"
RANKS="${RANKS:-1,2,4,8,16,64}"
MAX_LENGTH="${MAX_LENGTH:-384}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

extra=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  extra+=(--skip_existing)
fi

python projects/privacy_lora/run_audit.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --stage "${STAGE}" \
  --n_train "${N_TRAIN}" \
  --n_holdout "${N_HOLDOUT}" \
  --n_public "${N_PUBLIC}" \
  --n_mia "${N_MIA}" \
  --epochs "${EPOCHS}" \
  --ranks "${RANKS}" \
  --max_length "${MAX_LENGTH}" \
  "${extra[@]}"

python projects/privacy_lora/aggregate_to_tex.py \
  --root "${OUTPUT_DIR}" \
  --tex iclr2027_privacy_lora.tex

echo "Done. Results in ${OUTPUT_DIR}"
