# QuAda

Quantum-generated adapters for parameter-efficient fine-tuning of speech and
audio models (AST, WavLM, HuBERT).

A small parameterized quantum circuit and a shared MLP **generate** LoRA A/B
weights during training. After training the adapters are materialized and
merged into the frozen encoder, so **inference is fully classical**.


## Layout

```
mttl/models/modifiers/quada.py     # PQC, decoder, generator
mttl/models/modifiers/dora.py      # DoRA baseline
projects/quantum_adapters/
  finetune_quada.py                # train linear / LoRA / DoRA / QuAda
  wave_data.py                     # WavLM / HuBERT waveforms
tests/test_quada.py
```

Outputs go to `quada_output/<backbone>/<method>/<task>/metrics.json`.

## Setup

Use the `mttl` conda env on a **GPU node** (the env is ARM; login nodes are x86).

```bash
conda activate mttl
cd /path/to/mttl
export PYTHONPATH=.
```

## Tests

`tests/conftest.py` must not load (it pulls in unrelated LLM fixtures):

```bash
python -m pytest tests/test_quada.py --noconftest -s --tb=short
```

## Train one run

```bash
# parameter / qubit table (CPU, no data)
python projects/quantum_adapters/count_params.py

# AST + QuAda on ESC-50
python projects/quantum_adapters/finetune_quada.py \
    --task esc50 --backbone ast --method quada \
    --n_mlp 1024 --n_circuit_layers 8 \
    --output_dir quada_output/ast/quada/esc50

# baselines
python projects/quantum_adapters/finetune_quada.py \
    --task esc50 --backbone ast --method lora --lora_rank 4
python projects/quantum_adapters/finetune_quada.py \
    --task esc50 --backbone ast --method dora --lora_rank 4
python projects/quantum_adapters/finetune_quada.py \
    --task esc50 --backbone ast --method linear
```

Tasks: `esc50`, `ravdess`, `speech_commands`.  
Methods: `linear`, `lora`, `dora`, `quada`.  
Backbones: `ast`, `wavlm`, `hubert`.

WavLM automatically learns SUPERB layer weights; QuAda generates them too.


Or locally:

```bash
SMOKE=1 bash projects/quantum_adapters/run_experiments.sh
TASKS=esc50 METHODS=quada BACKBONE=ast bash projects/quantum_adapters/run_experiments.sh
```
