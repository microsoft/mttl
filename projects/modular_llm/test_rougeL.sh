#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=80000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p irlab --gres=gpu:h100:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10-00:05:00


export PYTHONPATH=../../
export WANDB_API_KEY=521486617982bd279e01673cb248d82ec595c935
export WANDB_PROJECT=train_sparse_moe
export AMLT_JOB_NAME=test_20tasks

export TASKS="wiqa_what_is_the_final_step_of_the_following_process,sciq_Multiple_Choice,adversarial_qa_droberta_answer_the_following_q,duorc_SelfRC_question_answering,cos_e_v1_11_description_question_option_id,wiki_qa_Is_This_True_,quail_description_context_question_text,wiki_hop_original_explain_relation,duorc_ParaphraseRC_build_story_around_qa,yelp_polarity_reviews_0_2_0,squad_v1_1_3_0_0,web_questions_potential_correct_answer,quoref_Found_Context_Online,quoref_Given_Context_Answer_Question,web_questions_get_the_answer,cot_sensemaking,wiki_qa_Topic_Prediction_Question_and_Answer_Pair,dbpedia_14_given_list_what_category_does_the_paragraph_belong_to,duorc_ParaphraseRC_extract_answer,super_glue_rte_1_0_2"

python eval_rougeL.py -k merge_or_route="base" finetune_task_name=$TASKS model=microsoft/Phi-3-mini-4k-instruct checkpoint=train_full_phi3/20_tasks/last.ckpt output_dir=phi3_test_rougel
