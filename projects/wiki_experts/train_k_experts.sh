
for task_name in "high_school_government_and_politics" "prehistory" "security_studies" "sociology" "college_biology"; do
    CUDA_VISIBLE_DEVICES=0 python train_experts.py \
    -c wiki-mmlu/gptneo_125m.json+wiki-mmlu/gptneo_125m_dense.json \
    -k output_dir=gptneo_125m_experts/${task_name}/ \
    finetune_task_name=${task_name} \
    num_train_epochs=10 \
    warmup_proportion=0.1 \
    tensorboard=True \
    precision=16-mixed
done

# if expert
CUDA_VISIBLE_DEVICES=0 python train_experts.py \
    -c wiki-mmlu/gptneo_125m.json+wiki-mmlu/gptneo_125m_dense.json \
    -k output_dir=gptneo_125m_experts/infollow/ \
    dataset=platypus \
    num_train_epochs=1 \
    warmup_proportion=0.1 \
    tensorboard=True \
    precision=16-mixed
