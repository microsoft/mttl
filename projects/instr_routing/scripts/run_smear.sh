

for temp in 0.75 0.5 0.25 1.; do
for center in 0.1 0.01 0.; do

OUTPUT_DIR=/tmp/runs/smear_temp${temp}_cent${center}/

/bin/rm -rf ${OUTPUT_DIR}

python finetune_llama.py \
    -c configs/platypus/gptneo_125m.json+configs/platypus/gptneo_125m_vsmear.json \
    -k output_dir=${OUTPUT_DIR} \
    eval_superni=False \
    eval_mmlu=True \
    n_skills=8 \
    total_steps=1000 \
    model_modifier=smear \
    router_temperature=${temp} \
    router_center_momentum=${center} \
    router_normalize_weights=True \
    tensorboard=True
    $*

done
done
