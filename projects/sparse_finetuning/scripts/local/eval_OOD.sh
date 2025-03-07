LIB_ID=None
EXP_NAME=None
MERGING_METHOD=uniform
CUDA_VISIBLE_DEVICES=0 python3 projects/sparse_finetuning/eval_library.py \
            -k \
            seed=42 \
            output_dir=eval/${EXP_NAME} \
            library_id=$LIB_ID \
            merge_or_route=${MERGING_METHOD} \
            include_task_source=* \
            dataset=sordonia/flan-10k-flat \
            predict_batch_size=4 \
            pipeline_eval_tasks=in_distribution \
            eval_metric=rougeL \
            finetune_task_name=glue_sst2_2_0_0,dream_read_the_following_conversation_and_answer_the_question,race_middle_Read_the_article_and_answer_the_question_no_option_,adversarial_qa_droberta_generate_question,adversarial_qa_dbidaf_question_context_answer,app_reviews_convert_to_star_rating,race_high_Select_the_best_answer,true_case,wiqa_what_might_be_the_first_step_of_the_process,quail_description_context_question_answer_id
            