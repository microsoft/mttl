/bin/rm -rf /tmp/dense/
python finetune_llama.py \
    -c configs/platypus/gptneo_125m.json+configs/platypus/gptneo_125m_dense.json \
    -k output_dir=/tmp/dense \
    eval_mmlu=True \
    eval_superni=True \
    total_steps=0
