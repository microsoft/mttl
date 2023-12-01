# flan tasks without NIv2 template
FLAN_SUB1 = [
    "ai2_arc_ARC_Challenge_1_0_0",
]

FLAN_SUB3 = [
    "ai2_arc_ARC_Challenge_1_0_0",
    "dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to",
    "web_questions_whats_the_answer",
]

FLAN_SUB19 = [
    "ai2_arc_ARC_Challenge_1_0_0",
    "dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to",
    "web_questions_whats_the_answer",
    "squad_v1_1_3_0_0",
    "social_i_qa_Check_if_a_random_answer_is_valid_or_not",
    "wiqa_what_might_be_the_first_step_of_the_process",
    "dbpedia_14_given_a_choice_of_categories_",
    "wiqa_effect_with_string_answer",
    "adversarial_qa_dbidaf_answer_the_following_q",
    "duorc_SelfRC_answer_question",
    "quoref_Find_Answer",
    "duorc_ParaphraseRC_answer_question",
    "duorc_ParaphraseRC_title_generation",
    "adversarial_qa_dbidaf_generate_question",
    "yelp_polarity_reviews_0_2_0",
    "dream_baseline",
    "cos_e_v1_11_question_description_option_text",
    "wiki_hop_original_choose_best_object_interrogative_2",
    "quartz_read_passage_below_choose",
]

FLAN_TASKS = [
    "ai2_arc_ARC_Challenge_1_0_0",
    "dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to",
    "web_questions_whats_the_answer",
    "squad_v1_1_3_0_0",
    "social_i_qa_Check_if_a_random_answer_is_valid_or_not",
    "wiqa_what_might_be_the_first_step_of_the_process",
    "dbpedia_14_given_a_choice_of_categories_",
    "wiqa_effect_with_string_answer",
    "adversarial_qa_dbidaf_answer_the_following_q",
    "duorc_SelfRC_answer_question",
    "quoref_Find_Answer",
    "duorc_ParaphraseRC_answer_question",
    "duorc_ParaphraseRC_title_generation",
    "cot_ecqa",
    "adversarial_qa_dbidaf_generate_question",
    "yelp_polarity_reviews_0_2_0",
    "dream_baseline",
    "cos_e_v1_11_question_description_option_text",
    "wiki_hop_original_choose_best_object_interrogative_2",
    "quartz_read_passage_below_choose",
    "snli_1_1_0",
    "true_case",
    "wiki_qa_Is_This_True_",
    "dream_answer_to_dialogue",
    "quoref_Answer_Test",
    "quail_context_question_description_answer_text",
    "duorc_ParaphraseRC_generate_question",
    "wiki_qa_Topic_Prediction_Question_and_Answer_Pair",
    "web_questions_get_the_answer",
    "adversarial_qa_dbidaf_tell_what_it_is",
    "gem_web_nlg_en_1_1_0",
    "cot_gsm8k_ii",
    "coqa_1_0_0",
    "ropes_prompt_beginning",
    "unified_qa_science_inst",
    "kilt_tasks_hotpotqa_formulate",
    "social_i_qa_Generate_answer",
    "glue_mrpc_2_0_0",
    "quail_context_description_question_answer_id",
    "quoref_Guess_Answer",
    "quail_description_context_question_answer_id",
    "ropes_background_new_situation_answer",
    "ropes_prompt_mix",
    "quac_1_0_0",
    "cot_ecqa_ii",
    "wiki_qa_Decide_good_answer",
    "quoref_Answer_Friend_Question",
    "quoref_What_Is_The_Answer",
    "app_reviews_convert_to_rating",
    "adversarial_qa_dbert_based_on",
    "cos_e_v1_11_aligned_with_common_sense",
    "adversarial_qa_dbert_generate_question",
    "duorc_ParaphraseRC_generate_question_by_answer",
    "super_glue_cb_1_0_2",
    "glue_qqp_2_0_0",
    "web_questions_question_answer",
    "openbookqa_0_1_0",
    "wiki_hop_original_choose_best_object_interrogative_1",
    "stream_qed",
    "cot_qasc_ii",
    "anli_r2_0_1_0",
    "word_segment",
    "wiki_hop_original_choose_best_object_affirmative_3",
    "quail_context_question_answer_description_text",
    "quartz_use_info_from_question_paragraph",
    "cos_e_v1_11_question_description_option_id",
    "duorc_ParaphraseRC_build_story_around_qa",
    "adversarial_qa_dbert_answer_the_following_q",
    "super_glue_wic_1_0_2",
    "web_questions_potential_correct_answer",
    "race_middle_Select_the_best_answer_no_instructions_",
    "app_reviews_generate_review",
    "quartz_answer_question_below",
    "wmt16_translate_ro_en_1_0_0",
    "wiki_hop_original_choose_best_object_affirmative_1",
    "super_glue_rte_1_0_2",
    "gem_common_gen_1_1_0",
    "wmt14_translate_fr_en_1_0_0",
    "kilt_tasks_hotpotqa_straighforward_qa",
    "cot_sensemaking",
    "app_reviews_convert_to_star_rating",
    "race_high_Write_a_multi_choice_question_options_given_",
    "cosmos_qa_1_0_0",
    "wiki_hop_original_choose_best_object_affirmative_2",
    "ropes_new_situation_background_answer",
    "duorc_ParaphraseRC_question_answering",
    "wiki_qa_Jeopardy_style",
    "ropes_plain_bottom_hint",
    "quail_no_prompt_id",
    "cos_e_v1_11_generate_explanation_given_text",
    "wiki_bio_guess_person",
    "winogrande_1_1_0",
    "adversarial_qa_dbert_question_context_answer",
    "stream_aqua",
    "quartz_paragraph_question_plain_concat",
    "kilt_tasks_hotpotqa_complex_question",
    "race_middle_Write_a_multi_choice_question_options_given_",
    "quarel_logic_test",
    "ropes_prompt_bottom_no_hint",
    "adversarial_qa_droberta_based_on",
    "qasc_qa_with_separated_facts_5",
    "sciq_Multiple_Choice_Closed_Book_",
    "sciq_Direct_Question_Closed_Book_",
    "duorc_SelfRC_generate_question_by_answer",
    "cot_strategyqa_ii",
    "wiki_hop_original_explain_relation",
    "cos_e_v1_11_description_question_option_id",
    "wmt16_translate_fi_en_1_0_0",
    "fix_punct",
    "quartz_given_the_fact_answer_the_q",
    "glue_sst2_2_0_0",
    "adversarial_qa_droberta_generate_question",
    "quail_context_question_description_text",
    "cot_creak",
    "cnn_dailymail_3_4_0",
    "wiki_hop_original_generate_subject",
    "cot_qasc",
    "ropes_read_background_situation",
    "huggingface_xsum",
    "quartz_use_info_from_paragraph_question",
    "qasc_qa_with_combined_facts_1",
    "wiki_qa_Direct_Answer_to_Question",
    "ropes_plain_no_background",
    "duorc_SelfRC_question_answering",
    "adversarial_qa_droberta_tell_what_it_is",
    "gem_e2e_nlg_1_1_0",
    "race_middle_Read_the_article_and_answer_the_question_no_option_",
    "gigaword_1_2_0",
    "quoref_Read_And_Extract_",
    "duorc_SelfRC_title_generation",
    "wiki_hop_original_generate_subject_and_object",
    "dream_generate_first_utterance",
    "glue_wnli_2_0_0",
    "wmt16_translate_de_en_1_0_0",
    "super_glue_copa_1_0_2",
    "adversarial_qa_droberta_answer_the_following_q",
    "qasc_qa_with_separated_facts_4",
    "wiki_bio_what_content",
    "wiki_qa_found_on_google",
    "kilt_tasks_hotpotqa_combining_facts",
    "math_dataset_algebra__linear_1d_1_0_0",
    "wiki_qa_Generate_Question_from_Topic",
    "wiki_qa_automatic_system",
    "race_middle_Select_the_best_answer",
    "ropes_plain_background_situation",
    "quarel_choose_between",
    "duorc_ParaphraseRC_extract_answer",
    "wiki_qa_Topic_Prediction_Question_Only",
    "kilt_tasks_hotpotqa_final_exam",
    "ag_news_subset_1_0_0",
    "wiki_qa_exercise",
    "adversarial_qa_dbidaf_based_on",
    "sciq_Multiple_Choice",
    "dream_read_the_following_conversation_and_answer_the_question",
    "wiqa_what_is_the_final_step_of_the_following_process",
    "wiqa_which_of_the_following_is_the_supposed_perturbation",
    "quail_context_description_question_text",
    "cos_e_v1_11_question_option_description_id",
    "adversarial_qa_dbidaf_question_context_answer",
    "trivia_qa_rc_1_1_0",
    "quail_no_prompt_text",
    "sciq_Direct_Question",
    "gem_wiki_lingua_english_en_1_1_0",
    "ropes_prompt_bottom_hint_beginning",
    "quoref_Answer_Question_Given_Context",
    "super_glue_wsc_fixed_1_0_2",
    "wiqa_effect_with_label_answer",
    "duorc_ParaphraseRC_decide_worth_it",
    "squad_v2_0_3_0_0",
    "piqa_1_0_0",
    "super_glue_record_1_0_2",
    "quoref_Context_Contains_Answer",
    "dbpedia_14_given_list_what_category_does_the_paragraph_belong_to",
    "duorc_SelfRC_build_story_around_qa",
    "quail_context_question_answer_description_id",
    "anli_r1_0_1_0",
    "quarel_heres_a_story",
    "stream_qed_ii",
    "race_high_Read_the_article_and_answer_the_question_no_option_",
    "natural_questions_open_1_0_0",
    "duorc_ParaphraseRC_movie_director",
    "quail_description_context_question_answer_text",
    "quartz_having_read_above_passage",
    "app_reviews_categorize_rating_using_review",
    "glue_stsb_2_0_0",
    "social_i_qa_Show_choices_and_generate_index",
    "cos_e_v1_11_description_question_option_text",
    "wiqa_does_the_supposed_perturbation_have_an_effect",
    "lambada_1_0_0",
    "hellaswag_1_1_0",
    "ropes_background_situation_middle",
    "glue_qnli_2_0_0",
    "qasc_is_correct_1",
    "quarel_testing_students",
    "wiqa_what_is_the_missing_first_step",
    "para_crawl_enes",
    "dbpedia_14_pick_one_category_for_the_following_text",
    "duorc_SelfRC_decide_worth_it",
    "cot_strategyqa",
    "wmt16_translate_tr_en_1_0_0",
    "aeslc_1_0_0",
    "cos_e_v1_11_rationale",
    "cos_e_v1_11_i_think",
    "race_middle_Taking_a_test",
    "quarel_do_not_use",
    "race_high_Is_this_the_right_answer",
    "race_middle_Write_a_multi_choice_question_for_the_following_article",
    "duorc_SelfRC_generate_question",
    "multi_news_1_0_0",
    "race_middle_Is_this_the_right_answer",
    "imdb_reviews_plain_text_1_0_0",
    "social_i_qa_Show_choices_and_generate_answer",
    "quail_context_question_description_answer_id",
    "cot_creak_ii",
    "gem_dart_1_1_0",
    "paws_wiki_1_1_0",
    "wiqa_what_might_be_the_last_step_of_the_process",
    "duorc_SelfRC_movie_director",
    "glue_cola_2_0_0",
    "bool_q_1_0_0",
    "drop_2_0_0",
    "cot_esnli_ii",
    "race_high_Select_the_best_answer",
    "cos_e_v1_11_question_option_description_text",
    "social_i_qa_Generate_the_question_from_the_answer",
    "wiki_bio_comprehension",
    "race_high_Select_the_best_answer_no_instructions_",
    "anli_r3_0_1_0",
    "wiki_bio_who",
    "web_questions_short_general_knowledge_q",
    "qasc_qa_with_separated_facts_3",
    "cos_e_v1_11_explain_why_human",
    "qasc_qa_with_separated_facts_2",
    "race_high_Taking_a_test",
    "qasc_is_correct_2",
    "wiki_hop_original_generate_object",
    "wiki_qa_Topic_Prediction_Answer_Only",
    "glue_mnli_2_0_0",
    "wiki_bio_key_content",
    "quoref_Guess_Title_For_Context",
    "definite_pronoun_resolution_1_1_0",
    "ropes_given_background_situation",
    "qasc_qa_with_separated_facts_1",
    "quoref_Found_Context_Online",
    "stream_aqua_ii",
    "cot_sensemaking_ii",
    "adversarial_qa_droberta_question_context_answer",
    "race_high_Select_the_best_answer_generate_span_",
    "ai2_arc_ARC_Easy_1_0_0",
    "adversarial_qa_dbert_tell_what_it_is",
    "cot_gsm8k",
    "quail_context_description_question_answer_text",
    "trec_1_0_0",
    "quartz_answer_question_based_on",
    "super_glue_multirc_1_0_2",
    "dream_generate_last_utterance",
    "sciq_Multiple_Choice_Question_First",
    "cot_esnli",
    "race_middle_Select_the_best_answer_generate_span_",
    "social_i_qa_I_was_wondering",
    "quoref_Given_Context_Answer_Question",
    "duorc_SelfRC_extract_answer",
    "quail_description_context_question_text",
    "race_high_Write_a_multi_choice_question_for_the_following_article",
]
