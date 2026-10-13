#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH -o privacy_lora_v2/logs/%x_%j.out
#SBATCH -e privacy_lora_v2/logs/%x_%j.err
#SBATCH -J ll_vis_super

cd "${SLURM_SUBMIT_DIR:-$HOME/code/mttl}"
set -euo pipefail
source projects/medmergebench/slurm_env.sh
export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false
export PRIVACY_LORA_VISION=privacy_lora_v2/vision_superposition
export PRIVACY_LORA_MEDMNIST=privacy_lora_v2/medmnist

# medmnist may be missing in the shared env; install to user site if needed
python - <<'PY'
try:
    import medmnist
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'medmnist', '-q'])
PY

DATASET="${1:?dataset}"
SEED="${2:?seed}"
python -m projects.privacy_lora.vision.superposition --dataset "$DATASET" --seed "$SEED"
