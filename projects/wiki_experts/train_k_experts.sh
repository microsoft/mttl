
for task_name in "high_school_government_and_politics" "prehistory" "security_studies" "sociology" "college_biology"; do
    CUDA_VISIBLE_DEVICES=0 python train_experts.py \
    -c wiki-mmlu/gptneo_125m.json+wiki-mmlu/gptneo_125m_dense.json \
    -k output_dir=gptneo_125m_experts/${task_name}/ \
    finetune_task_name=${task_name} \
    warmup_proportion=0.1
done
