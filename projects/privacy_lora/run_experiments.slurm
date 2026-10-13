#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o privacy_lora_%j.out
#SBATCH -e privacy_lora_%j.err
#SBATCH -J LoRALeak

cd "${SLURM_SUBMIT_DIR:-$HOME/code/mttl}"
set -euo pipefail
source projects/medmergebench/slurm_env.sh

bash projects/privacy_lora/run_experiments.sh
