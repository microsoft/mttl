/bin/rm -rf /tmp/runs/vsmear/

python finetune_llama.py \
    -c configs/platypus/gptneo_125m.json+configs/platypus/gptneo_125m_dense.json \
    -k output_dir=/tmp/runs/vsmear/ \
    eval_superni=True \
    eval_mmlu=True \
    n_skills=8 \
    model_modifier=vsmear \
    total_steps=1000
