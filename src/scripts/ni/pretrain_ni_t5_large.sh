if [ $# -lt 4 ]
  then
      echo "Missing arguments. Please call: 'sh pretrain_ni_t5_large.sh <exp_name> <train_dir> <output_dir>'"
      exit 1
fi
exp_name=$1
train_dir=$2
output_dir=$3

python pl_train.py \
    --exp_name "$exp_name" \
    --train_dir "$train_dir" \
    --output_dir "$output_dir" \
    --custom_tasks_splits "${train_dir}/splits/default/train_tasks.txt" \
    --model t5-large \
    --train_batch_size 8 \
    --predict_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --warmup_steps 1569 \
    --learning_rate 5e-5 \
    --max_input_length 1024 \
    --max_output_length 128 \
    --finetune_full_model \
    --use_task_descriptions \
    --precision bf16 \
    --dataset ni
