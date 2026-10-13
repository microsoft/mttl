#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH -o privacy_lora_v2/logs/%x_%j.out
#SBATCH -e privacy_lora_v2/logs/%x_%j.err
#SBATCH -J loraleak

# usage: sbatch -J NAME projects/privacy_lora/v2/run_arms.slurm SPEC_FILE [ONLY]
cd "${SLURM_SUBMIT_DIR:-$HOME/code/mttl}"
set -euo pipefail
source projects/medmergebench/slurm_env.sh
export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
SPEC=$1
ONLY=${2:-}
if [ -n "$ONLY" ]; then
  python -m projects.privacy_lora.v2.run_arm --spec_file "$SPEC" --only "$ONLY"
else
  python -m projects.privacy_lora.v2.run_arm --spec_file "$SPEC"
fi
