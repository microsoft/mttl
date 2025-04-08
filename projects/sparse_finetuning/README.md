


* Execute the following command to train locally
```
projects/sparse_finetuning/scripts/local/run.sh "full" "local://temp/test_library"
```
* Argument Options:
    * (1) criteria of training
        * `regular_sparse`: regular-sparse 
        * `block_sparse`: block-sparse
    * (2) library location
        * `local://<dir_location>` to save locally
        * `hf://<user_id>` to upload to huggingface

---
* List of tasks to train: 
        `
        ['wiqa_what_is_the_final_step_of_the_following_process', 'sciq_Multiple_Choice', 'adversarial_qa_droberta_answer_the_following_q', 'duorc_SelfRC_question_answering', 'cos_e_v1_11_description_question_option_id', 'wiki_qa_Is_This_True_', 'quail_description_context_question_text', 'wiki_hop_original_explain_relation', 'duorc_ParaphraseRC_build_story_around_qa', 'yelp_polarity_reviews_0_2_0', 'squad_v1_1_3_0_0', 'web_questions_potential_correct_answer', 'quoref_Found_Context_Online', 'quoref_Given_Context_Answer_Question', 'web_questions_get_the_answer', 'cot_sensemaking', 'wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to', 'duorc_ParaphraseRC_extract_answer', 'super_glue_rte_1_0_2']
        `
        
---
* Notes:
    * conda create -n mttl python=3.9
    * conda activate mttl
    * pip install -r local_requirements.txt
    * use `transformers==4.42.0` to properly use local `microsoft/phi-2` model
