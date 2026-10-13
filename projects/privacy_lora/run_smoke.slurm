#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o privacy_lora_smoke_%j.out
#SBATCH -e privacy_lora_smoke_%j.err
#SBATCH -J LoRALeakSmoke

cd "${SLURM_SUBMIT_DIR:-$HOME/code/mttl}"
set -euo pipefail
source projects/medmergebench/slurm_env.sh

SMOKE=1 MODEL_NAME=meta-llama/Llama-3.2-1B \
  OUTPUT_DIR=privacy_lora_artifacts/smoke \
  bash projects/privacy_lora/run_experiments.sh
