#!/bin/bash
# 8 tasks
task_list="wiqa_what_is_the_final_step_of_the_following_process sciq_Multiple_Choice adversarial_qa_droberta_answer_the_following_q duorc_SelfRC_question_answering cos_e_v1_11_description_question_option_id wiki_qa_Is_This_True_ quail_description_context_question_text wiki_hop_original_explain_relation"

for item in $task_list
do
  echo "Processing: $item"
  python projects/modular_llm/train_experts.py -c projects/modular_llm/configs/models/gptneo_125m.json -k num_train_epochs=1 micro_batch_size=4 finetune_task_name=$item total_steps=100  dataset=sordonia/flan-debug-flat library_id=zhan1993/trained_gpt125m_experts_colab
done