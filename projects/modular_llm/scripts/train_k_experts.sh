
for task_name in "high_school_government_and_politics"; do
    python train_experts.py \
    -c wiki-mmlu/llama2_13b_experts+wiki-mmlu/llama2_13b_dense \
    -k output_dir=results/${task_name}/ \
    expert_name=${task_name} \
    finetune_task_name=${task_name} \
    tensorboard=True
done