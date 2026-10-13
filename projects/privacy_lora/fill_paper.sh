#!/bin/bash
# After the full 8B job finishes, refresh the paper tables from
# privacy_lora_artifacts/ (not the smoke overlay).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"

python projects/privacy_lora/aggregate_to_tex.py \
  --root privacy_lora_artifacts \
  --tex iclr2027_privacy_lora.tex \
  --print_rep | tee privacy_lora_artifacts/last_tex_fill.json

echo "Updated iclr2027_privacy_lora.tex from privacy_lora_artifacts/"
