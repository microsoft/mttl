/bin/rm -rf /tmp/runs/vsmear/

python finetune_llama.py \
    -c configs/platypus/gptneo_125m.json+configs/platypus/gptneo_125m_vsmear.json \
    -k output_dir=/tmp/runs/vsmear/ \
    eval_superni=True \
    eval_mmlu=True \
    n_skills=8 \
    total_steps=1000 \
    $*
