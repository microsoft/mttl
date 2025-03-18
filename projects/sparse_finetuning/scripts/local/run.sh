#!/bin/bash


# EXP_NAME : ["full","lora","regular_sparse","block_sparse"]
EXP_NAME=$1
LIB_ID=$2

export WANDB_PROJECT="test"
declare -a task_name_list=('duorc_SelfRC_question_answering') # 'duorc_SelfRC_question_answering')

for task_name in "${task_name_list[@]}"; do
    export JOB_NAME="qwen_${EXP_NAME}/${task_name}"
    
    CUDA_VISIBLE_DEVICES=0 python3 projects/sparse_finetuning/train_experts_main.py \
    -c "projects/sparse_finetuning/configs/qwen_test_sparse.json" \
    -k \
    seed=42 \
    finetune_task_name=${task_name} \
    expert_name=${task_name} \
    wandb_project=${WANDB_PROJECT} \
    library_id=${LIB_ID}/${EXP_NAME} \
    output_dir=./experiment/${JOB_NAME}

    sleep 1
done

    # -c "projects/sparse_finetuning/configs/phi-3_flan_${EXP_NAME}.json" \
