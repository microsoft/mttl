OUTPUT_DIR=/tmp/runs/smear/

/bin/rm -rf ${OUTPUT_DIR}

python finetune_llama.py \
    -c configs/platypus/gptneo_125m.json+configs/platypus/gptneo_125m_vsmear.json \
    -k output_dir=${OUTPUT_DIR} \
    eval_superni=True \
    eval_mmlu=True \
    n_skills=8 \
    total_steps=1000 \
    model_modifier=smear
