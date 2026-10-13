#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH -o privacy_lora_sgd_%j.out
#SBATCH -e privacy_lora_sgd_%j.err
#SBATCH -J LoRALeakSGD

cd "${SLURM_SUBMIT_DIR:-$HOME/code/mttl}"
set -euo pipefail
source projects/medmergebench/slurm_env.sh
export PYTHONPATH=./
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import json, os, shutil, sys
sys.path.insert(0, ".")
from projects.privacy_lora.clinical_data import load_corpus
from projects.privacy_lora.train_adapter import TrainConfig, train_lora_on_notes, load_adapted_model
from projects.privacy_lora.attacks import (
    attack_a1_base_calibrated, attack_a2_subspace, attack_loss_threshold
)

out = "privacy_lora_artifacts/sgd_ablation"
os.makedirs(out, exist_ok=True)
src_corpus = "privacy_lora_artifacts/corpus.json"
shutil.copy(src_corpus, os.path.join(out, "corpus.json"))
corpus = load_corpus(os.path.join(out, "corpus.json"))

cfg = TrainConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    lora_rank=16, lora_alpha=16.0,
    num_train_epochs=3, batch_size=1, grad_accum=8,
    max_length=384, learning_rate=1e-3, optimizer="sgd",
)
path = train_lora_on_notes(
    corpus["train"], cfg, os.path.join(out, "adapters/sgd_r16")
)["adapter_path"]
device = "cuda"
model, tok, _, meta = load_adapted_model(cfg.model_name, path, device)
members, nonmembers = corpus["train"][:200], corpus["holdout"][:200]
# Compare to existing AdamW mia_main.json
adam = json.load(open("privacy_lora_artifacts/mia_main.json"))
rows = {
    "sgd": {
        "loss": attack_loss_threshold(model, tok, members, nonmembers, 384, device),
        "a1": attack_a1_base_calibrated(model, tok, members, nonmembers, 384, device),
        "a2": attack_a2_subspace(model, tok, members[:80], nonmembers[:80], 384, device),
    },
    "adamw_reference": {
        "loss": adam["loss_threshold"],
        "a1": adam["a1"],
        "a2": adam["a2"],
    },
}
path_out = os.path.join(out, "sgd_vs_adam_mia.json")
with open(path_out, "w") as f:
    json.dump(rows, f, indent=2)
print(json.dumps(rows, indent=2))
print("wrote", path_out)
PY
