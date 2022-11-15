if [ $# -lt 4 ]
  then
      echo "Missing arguments. Please call: 'sh finetune_ni.sh <exp_name> <train_dir> <output_dir> <checkpoint_dir>'"
      exit 1
fi
exp_name=$1
train_dir=$2
output_dir=$3
checkpoint_dir=$4

tasks=(
    "task1157_bard_analogical_reasoning_rooms_for_containers"
    "task1358_xlsum_title_generation"
    "task677_ollie_sentence_answer_generation"
    "task1385_anli_r1_entailment"
    "task1152_bard_analogical_reasoning_causation"
    "task304_numeric_fused_head_resolution"
    "task671_ambigqa_text_generation"
    "task880_schema_guided_dstc8_classification"
    "task1161_coda19_title_generation"
    "task1624_disfl_qa_question_yesno_classification"
    "task1598_nyc_long_text_generation"
    "task201_mnli_neutral_classification"
    "task233_iirc_link_exists_classification"
    "task035_winogrande_question_modification_person"
    "task957_e2e_nlg_text_generation_generate"
    "task1356_xlsum_title_generation"
    "task1531_daily_dialog_type_classification"
    "task1154_bard_analogical_reasoning_travel"
    "task1622_disfl_qa_text_modication"
    "task1393_superglue_copa_text_completion"
)

for i in "${tasks[@]}"; do
    python pl_finetune.py \
        --exp_name "${exp_name}_${i}" \
        --train_dir "$train_dir" \
        --output_dir "$output_dir" \
        --checkpoint "$checkpoint_dir" \
        --finetune_task_name "$i" \
        --dataset ni \
        --model t5-large \
        --warmup_steps 100 \
        --num_train_epochs 100 \
        --train_batch_size 2 \
        --gradient_accumulation_steps 2 \
        --predict_batch_size 8 \
        --precision 32
done
