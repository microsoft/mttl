/bin/rm -rf /tmp/runs/dense/
python finetune_llama.py \
    -c configs/platypus/gptneo_125m.json+configs/platypus/gptneo_125m_dense.json \
    -k output_dir=/tmp/runs/dense/ \
    eval_mmlu=True \
    eval_superni=False \
    total_steps=1000 \
    $*
