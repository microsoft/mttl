#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH -o privacy_lora_v2/logs/%x_%j.out
#SBATCH -e privacy_lora_v2/logs/%x_%j.err
#SBATCH -J ll_vis_mia

cd "${SLURM_SUBMIT_DIR:-$HOME/code/mttl}"
set -euo pipefail
source projects/medmergebench/slurm_env.sh
export PYTHONPATH=./
export PRIVACY_LORA_VISION_MIA=privacy_lora_v2/vision_mia
export PRIVACY_LORA_MEDMNIST=privacy_lora_v2/medmnist
python - <<'PY'
try:
    import medmnist
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'medmnist', '-q'])
try:
    import sklearn
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn', '-q'])
PY
python -m projects.privacy_lora.vision.mia_vision --dataset "$1" --seed "$2"
