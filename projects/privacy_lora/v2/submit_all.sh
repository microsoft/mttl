#!/bin/bash
# Submit one Slurm job per spec file. usage: bash submit_all.sh SPEC_DIR [pattern]
set -euo pipefail
cd "$HOME/code/mttl"
DIR=${1:-privacy_lora_v2/specs}
PAT=${2:-*}
for f in "$DIR"/$PAT.jsonl; do
  name=$(basename "$f" .jsonl)
  [ "$name" = "smoke" ] && continue
  jid=$(sbatch --parsable -J "ll_$name" projects/privacy_lora/v2/run_arms.slurm "$f")
  echo "$name $jid"
done
