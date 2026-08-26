


source /nobackup/proj/disk/rare-disease-llm/shared/miniconda3/etc/profile.d/conda.sh
conda activate mttl
cd ~/code/mttl

export PYTHONPATH=./
python projects/modular_llm/eval_library.py -k library_id=hf://zhan1993/private_library_phi3-4k predict_batch_size=4 device_map=auto merge_or_route=osrm expert_selection=wiqa_what_is_the_final_step_of_the_following_process,sciq_Multiple_Choice,adversarial_qa_droberta_answer_the_following_q,duorc_SelfRC_question_answering,cos_e_v1_11_description_question_option_id,wiki_qa_Is_This_True_,quail_description_context_question_text,wiki_hop_original_explain_relation,duorc_ParaphraseRC_build_story_around_qa,yelp_polarity_reviews_0_2_0 pipeline_eval_tasks=in_distribution eval_metric=rougeL

# MedMergeBench (after train_nlp_experts.sh):
# python projects/modular_llm/eval_library.py -k \
#   library_id=local://medmergebench_artifacts/nlp_library \
#   dataset=local://medmergebench_artifacts/nlp_flat \
#   merge_or_route=dare_ties \
#   expert_selection=medqa,medmcqa,pubmedqa,bioasq,n2c2_medrec,i2b2_assertion,clicr,casi \
#   pipeline_eval_tasks=in_distribution eval_metric=rougeL predict_batch_size=4 device_map=auto
# Paper metrics (accuracy / F1 / Row-EM) are in:
#   bash projects/medmergebench/eval_merging.sh

