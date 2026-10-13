#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH -o privacy_lora_v2/logs/%x_%j.out
#SBATCH -e privacy_lora_v2/logs/%x_%j.err
#SBATCH -J ll_super_q14

cd "${SLURM_SUBMIT_DIR:-$HOME/code/mttl}"
set -euo pipefail
source projects/medmergebench/slurm_env.sh
export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false
export PRIVACY_LORA_SUPERPOS=privacy_lora_v2/superposition_v3

MODEL="${1:?model}"
DATASET="${2:?dataset}"
SEED="${3:?seed}"
python -m projects.privacy_lora.v2.superposition --model "$MODEL" --dataset "$DATASET" --seed "$SEED"
