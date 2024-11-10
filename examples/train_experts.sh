#!/bin/bash

TASKS=(
    "wiqa_what_is_the_final_step_of_the_following_process"
    "sciq_Multiple_Choice"
    "adversarial_qa_droberta_answer_the_following_q"
    "duorc_SelfRC_question_answering"
    "cos_e_v1_11_description_question_option_id"
    "wiki_qa_Is_This_True_"
    "quail_description_context_question_text"
    "wiki_hop_original_explain_relation"
)

for TASK in "${TASKS[@]}"
do
  echo "Processing: $TASK"
  python projects/modular_llm/train_experts.py -c projects/modular_llm/configs/models/gptneo_125m_fast.json \
  -k finetune_task_name=$TASK \
  -k library_id=local://trained_gpt125m_experts_colab
done
