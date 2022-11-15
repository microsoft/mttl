if [ $# -lt 3 ]
  then
      echo "Missing arguments. Please call: 'sh polytropon_ni_poly_8x8.sh <exp_name> <train_dir> <output_dir>'"
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
    --predict_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --warmup_steps 1569 \
    --total_steps 26156 \
    --learning_rate 5e-5 \
    --module_logits_learning_rate 0.1 \
    --max_input_length 1024 \
    --max_output_length 128 \
    --use_task_descriptions \
    --precision bf16 \
    --dataset ni \
    --selector polytropon \
    --n_skills 8 \
    --n_splits 8 \
    --finegrained
