if [ $# -lt 3 ]
  then
      echo "Missing arguments. Please call: 'sh polytropon_xfit.sh <exp_name> <train_dir> <output_dir>'"
      exit 1
fi
exp_name=$1
train_dir=$2
output_dir=$3

python pl_train.py \
    --exp_name "$exp_name" \
    --train_dir "$train_dir" \
    --output_dir "$output_dir" \
    --custom_tasks_splits dataloader/xfit_data/random.json \
    --dataset xfit \
    --model facebook/bart-large \
    --train_batch_size 8 \
    --predict_batch_size 16 \
    --num_train_epochs 30 \
    --n_skills 8 \
    --n_splits 8 \
    --finegrained
