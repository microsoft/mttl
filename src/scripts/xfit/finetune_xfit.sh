if [ $# -lt 4 ]
  then
      echo "Missing arguments. Please call: 'sh finetune_xfit.sh <exp_name> <train_dir> <output_dir> <checkpoint_dir>'"
      exit 1
fi
exp_name=$1
train_dir=$2
output_dir=$3
checkpoint_dir=$4

tasks=(
    "ag_news"
    "ai2_arc"
    "quoref"
    "wiki_split"
    "ethos-disability"
    "yelp_polarity"
    "superglue-rte"
    "glue-cola"
    "ethos-sexual_orientation"
    "blimp-sentential_negation_npi_scope"
    "amazon_polarity"
    "race-high"
    "blimp-sentential_negation_npi_licensor_present"
    "tweet_eval-irony"
    "break-QDMR"
    "crawl_domain"
    "freebase_qa"
    "glue-qnli"
    "hatexplain"
    "circa"
)

for i in "${tasks[@]}"; do
    python pl_finetune.py \
        --exp_name "${exp_name}_${i}" \
        --train_dir "$train_dir" \
        --output_dir "$output_dir" \
        --checkpoint "$checkpoint_dir" \
        --finetune_task_name "$i" \
        --dataset xfit \
        --warmup_proportion 0.1 \
        --total_steps 1000 \
        --train_batch_size 8 \
        --predict_batch_size 8 \
        --precision 32
done
