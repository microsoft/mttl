# Trains a private library of experts for Phi-2

if [ -z "${LIBRARY_PATH}" ]; then
    echo "Error: LIBRARY_PATH is not set, must be of the form hf://username/library_name, or local://path/to/library"
    exit 1
fi

if [ -z "${DATASET_PATH}" ]; then
    echo "Error: DATASET_PATH is not set, must be a valid dataset on huggingface, create the dataset with cli_dataset_create.py"
    exit 1
fi

for task_name in "race_middle_Select_the_best_answer_no_instructions_"  "drop_2_0_0"; do
    CUDA_VISIBLE_DEVICES=0 python train_experts_main.py \
        -c configs/models/gptneo_125m.json \
        -k \
        output_dir=output/${task_name}/ \
        finetune_task_name=${task_name} \
        expert_name=${task_name} \
        dataset=$DATASET_PATH \
        num_train_epochs=1 \
        library_id=$LIBRARY_PATH
done
