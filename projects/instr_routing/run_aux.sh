/bin/rm -rf /tmp/aux/
python finetune_llama.py \
    -c configs/platypus/gptneo_125m.json+configs/platypus/gptneo_125m_dense.json \
    -k output_dir=/tmp/ \
    eval_superni=True \
    n_skills=8 \
    model_modifier=aux_lora \
    router_selector=aux_var_router \
    total_steps=100
