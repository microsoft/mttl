from dataclasses import dataclass
from mttl.config import Config
import os
import json

tasks_names_to_ids = {
    "web_questions_question_answer": 0,
    "cos_e_v1_11_question_option_description_id": 1,
    "squad_v1_1_3_0_0": 2,
    "wmt14_translate_fr_en_1_0_0": 3,
    "cos_e_v1_11_explain_why_human": 4,
    "imdb_reviews_plain_text_1_0_0": 5,
    "wiki_bio_what_content": 6,
    "gem_wiki_lingua_english_en_1_1_0": 7,
    "anli_r1_0_1_0": 8,
    "super_glue_multirc_1_0_2": 9,
    "quartz_answer_question_based_on": 10,
    "adversarial_qa_dbidaf_generate_question": 11,
    "ropes_background_situation_middle": 12,
    "race_high_Select_the_best_answer_no_instructions_": 13,
    "race_high_Write_a_multi_choice_question_options_given_": 14,
    "quail_description_context_question_text": 15,
    "adversarial_qa_droberta_generate_question": 16,
    "ropes_plain_bottom_hint": 17,
    "wiki_hop_original_choose_best_object_interrogative_1": 18,
    "qasc_qa_with_combined_facts_1": 19,
    "dream_read_the_following_conversation_and_answer_the_question": 20,
    "dbpedia_14_given_a_choice_of_categories_": 21,
    "true_case": 22,
    "cosmos_qa_1_0_0": 23,
    "multi_news_1_0_0": 24,
    "app_reviews_categorize_rating_using_review": 25,
    "cos_e_v1_11_rationale": 26,
    "adversarial_qa_dbert_answer_the_following_q": 27,
    "race_middle_Select_the_best_answer_no_instructions_": 28,
    "wiki_qa_automatic_system": 29,
    "cnn_dailymail_3_4_0": 30,
    "quoref_Given_Context_Answer_Question": 31,
    "wiqa_effect_with_string_answer": 32,
    "race_high_Is_this_the_right_answer": 33,
    "quail_context_question_answer_description_text": 34,
    "web_questions_short_general_knowledge_q": 35,
    "cos_e_v1_11_aligned_with_common_sense": 36,
    "qasc_qa_with_separated_facts_3": 37,
    "ai2_arc_ARC_Challenge_1_0_0": 38,
    "gem_web_nlg_en_1_1_0": 39,
    "quarel_heres_a_story": 40,
    "math_dataset_algebra__linear_1d_1_0_0": 41,
    "quarel_logic_test": 42,
    "dream_answer_to_dialogue": 43,
    "ropes_plain_background_situation": 44,
    "anli_r2_0_1_0": 45,
    "super_glue_copa_1_0_2": 46,
    "duorc_ParaphraseRC_answer_question": 47,
    "social_i_qa_I_was_wondering": 48,
    "wiki_qa_Decide_good_answer": 49,
    "qasc_qa_with_separated_facts_4": 50,
    "glue_wnli_2_0_0": 51,
    "wiki_hop_original_choose_best_object_affirmative_1": 52,
    "wiki_qa_Topic_Prediction_Answer_Only": 53,
    "cos_e_v1_11_description_question_option_id": 54,
    "race_middle_Write_a_multi_choice_question_for_the_following_article": 55,
    "duorc_ParaphraseRC_question_answering": 56,
    "wmt16_translate_ro_en_1_0_0": 57,
    "sciq_Multiple_Choice_Closed_Book_": 58,
    "gem_dart_1_1_0": 59,
    "unified_qa_science_inst": 60,
    "duorc_ParaphraseRC_generate_question_by_answer": 61,
    "social_i_qa_Check_if_a_random_answer_is_valid_or_not": 62,
    "glue_stsb_2_0_0": 63,
    "ropes_prompt_mix": 64,
    "race_high_Write_a_multi_choice_question_for_the_following_article": 65,
    "super_glue_wic_1_0_2": 66,
    "super_glue_rte_1_0_2": 67,
    "app_reviews_convert_to_star_rating": 68,
    "dream_generate_first_utterance": 69,
    "quac_1_0_0": 70,
    "ropes_prompt_beginning": 71,
    "quoref_Answer_Friend_Question": 72,
    "wiki_qa_Is_This_True_": 73,
    "duorc_SelfRC_build_story_around_qa": 74,
    "duorc_ParaphraseRC_title_generation": 75,
    "web_questions_get_the_answer": 76,
    "duorc_SelfRC_decide_worth_it": 77,
    "wiki_hop_original_generate_subject_and_object": 78,
    "wiki_hop_original_generate_subject": 79,
    "quartz_having_read_above_passage": 80,
    "coqa_1_0_0": 81,
    "adversarial_qa_dbert_based_on": 82,
    "duorc_ParaphraseRC_movie_director": 83,
    "lambada_1_0_0": 84,
    "qasc_qa_with_separated_facts_1": 85,
    "word_segment": 86,
    "wiki_hop_original_generate_object": 87,
    "app_reviews_generate_review": 88,
    "quoref_Find_Answer": 89,
    "race_high_Select_the_best_answer_generate_span_": 90,
    "duorc_ParaphraseRC_generate_question": 91,
    "dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to": 92,
    "ropes_given_background_situation": 93,
    "quoref_Found_Context_Online": 94,
    "bool_q_1_0_0": 95,
    "social_i_qa_Show_choices_and_generate_index": 96,
    "quail_description_context_question_answer_id": 97,
    "drop_2_0_0": 98,
    "dream_generate_last_utterance": 99,
    "quoref_Answer_Question_Given_Context": 100,
    "race_high_Read_the_article_and_answer_the_question_no_option_": 101,
    "duorc_ParaphraseRC_build_story_around_qa": 102,
    "app_reviews_convert_to_rating": 103,
    "wiki_hop_original_choose_best_object_affirmative_3": 104,
    "gem_common_gen_1_1_0": 105,
    "quail_no_prompt_text": 106,
    "piqa_1_0_0": 107,
    "social_i_qa_Generate_the_question_from_the_answer": 108,
    "trivia_qa_rc_1_1_0": 109,
    "paws_wiki_1_1_0": 110,
    "wiqa_what_might_be_the_first_step_of_the_process": 111,
    "quarel_testing_students": 112,
    "wiki_hop_original_explain_relation": 113,
    "duorc_SelfRC_extract_answer": 114,
    "super_glue_record_1_0_2": 115,
    "wiki_bio_who": 116,
    "kilt_tasks_hotpotqa_formulate": 117,
    "quail_context_description_question_answer_id": 118,
    "sciq_Multiple_Choice_Question_First": 119,
    "para_crawl_enes": 120,
    "adversarial_qa_dbert_generate_question": 121,
    "qasc_qa_with_separated_facts_5": 122,
    "quartz_given_the_fact_answer_the_q": 123,
    "super_glue_cb_1_0_2": 124,
    "wiki_qa_Generate_Question_from_Topic": 125,
    "ropes_plain_no_background": 126,
    "sciq_Direct_Question_Closed_Book_": 127,
    "kilt_tasks_hotpotqa_complex_question": 128,
    "quartz_use_info_from_question_paragraph": 129,
    "glue_qqp_2_0_0": 130,
    "duorc_ParaphraseRC_decide_worth_it": 131,
    "quail_context_description_question_answer_text": 132,
    "huggingface_xsum": 133,
    "kilt_tasks_hotpotqa_final_exam": 134,
    "wiki_bio_guess_person": 135,
    "social_i_qa_Generate_answer": 136,
    "glue_qnli_2_0_0": 137,
    "cos_e_v1_11_i_think": 138,
    "kilt_tasks_hotpotqa_straighforward_qa": 139,
    "quartz_answer_question_below": 140,
    "quartz_read_passage_below_choose": 141,
    "trec_1_0_0": 142,
    "quail_context_question_description_answer_text": 143,
    "wiki_bio_comprehension": 144,
    "cos_e_v1_11_question_description_option_id": 145,
    "adversarial_qa_dbidaf_tell_what_it_is": 146,
    "quail_context_question_description_answer_id": 147,
    "cos_e_v1_11_question_description_option_text": 148,
    "quail_description_context_question_answer_text": 149,
    "natural_questions_open_1_0_0": 150,
    "web_questions_whats_the_answer": 151,
    "wiki_qa_Jeopardy_style": 152,
    "openbookqa_0_1_0": 153,
    "wiqa_what_is_the_final_step_of_the_following_process": 154,
    "wiki_qa_Direct_Answer_to_Question": 155,
    "snli_1_1_0": 156,
    "sciq_Multiple_Choice": 157,
    "cos_e_v1_11_generate_explanation_given_text": 158,
    "aeslc_1_0_0": 159,
    "adversarial_qa_dbert_tell_what_it_is": 160,
    "ai2_arc_ARC_Easy_1_0_0": 161,
    "race_high_Taking_a_test": 162,
    "quail_context_question_answer_description_id": 163,
    "wiqa_what_might_be_the_last_step_of_the_process": 164,
    "wiqa_effect_with_label_answer": 165,
    "adversarial_qa_dbidaf_answer_the_following_q": 166,
    "glue_mrpc_2_0_0": 167,
    "race_middle_Is_this_the_right_answer": 168,
    "wiqa_which_of_the_following_is_the_supposed_perturbation": 169,
    "wiki_qa_Topic_Prediction_Question_and_Answer_Pair": 170,
    "definite_pronoun_resolution_1_1_0": 171,
    "adversarial_qa_droberta_answer_the_following_q": 172,
    "duorc_SelfRC_answer_question": 173,
    "web_questions_potential_correct_answer": 174,
    "duorc_SelfRC_generate_question_by_answer": 175,
    "wiki_qa_exercise": 176,
    "quail_context_description_question_text": 177,
    "ropes_prompt_bottom_no_hint": 178,
    "duorc_SelfRC_title_generation": 179,
    "qasc_is_correct_1": 180,
    "wiki_bio_key_content": 181,
    "dream_baseline": 182,
    "glue_sst2_2_0_0": 183,
    "wiqa_what_is_the_missing_first_step": 184,
    "ag_news_subset_1_0_0": 185,
    "quarel_do_not_use": 186,
    "dbpedia_14_given_list_what_category_does_the_paragraph_belong_to": 187,
    "duorc_SelfRC_generate_question": 188,
    "anli_r3_0_1_0": 189,
    "wiki_hop_original_choose_best_object_affirmative_2": 190,
    "gem_e2e_nlg_1_1_0": 191,
    "adversarial_qa_droberta_based_on": 192,
    "cos_e_v1_11_question_option_description_text": 193,
    "social_i_qa_Show_choices_and_generate_answer": 194,
    "race_middle_Write_a_multi_choice_question_options_given_": 195,
    "quail_no_prompt_id": 196,
    "ropes_prompt_bottom_hint_beginning": 197,
    "squad_v2_0_3_0_0": 198,
    "sciq_Direct_Question": 199,
    "kilt_tasks_hotpotqa_combining_facts": 200,
    "ropes_new_situation_background_answer": 201,
    "quoref_Guess_Answer": 202,
    "gigaword_1_2_0": 203,
    "adversarial_qa_droberta_question_context_answer": 204,
    "wiqa_does_the_supposed_perturbation_have_an_effect": 205,
    "wiki_qa_Topic_Prediction_Question_Only": 206,
    "duorc_SelfRC_movie_director": 207,
    "quail_context_question_description_text": 208,
    "fix_punct": 209,
    "race_middle_Select_the_best_answer": 210,
    "wiki_qa_found_on_google": 211,
    "wmt16_translate_tr_en_1_0_0": 212,
    "wiki_hop_original_choose_best_object_interrogative_2": 213,
    "glue_mnli_2_0_0": 214,
    "quoref_Guess_Title_For_Context": 215,
    "duorc_SelfRC_question_answering": 216,
    "duorc_ParaphraseRC_extract_answer": 217,
    "wmt16_translate_de_en_1_0_0": 218,
    "quoref_Answer_Test": 219,
    "super_glue_wsc_fixed_1_0_2": 220,
    "quartz_paragraph_question_plain_concat": 221,
    "adversarial_qa_dbert_question_context_answer": 222,
    "dbpedia_14_pick_one_category_for_the_following_text": 223,
    "race_middle_Select_the_best_answer_generate_span_": 224,
    "quoref_What_Is_The_Answer": 225,
    "cos_e_v1_11_description_question_option_text": 226,
    "quoref_Read_And_Extract_": 227,
    "wmt16_translate_fi_en_1_0_0": 228,
    "adversarial_qa_dbidaf_question_context_answer": 229,
    "hellaswag_1_1_0": 230,
    "qasc_qa_with_separated_facts_2": 231,
    "winogrande_1_1_0": 232,
    "glue_cola_2_0_0": 233,
    "race_high_Select_the_best_answer": 234,
    "ropes_read_background_situation": 235,
    "race_middle_Read_the_article_and_answer_the_question_no_option_": 236,
    "quoref_Context_Contains_Answer": 237,
    "quarel_choose_between": 238,
    "ropes_background_new_situation_answer": 239,
    "quartz_use_info_from_paragraph_question": 240,
    "adversarial_qa_dbidaf_based_on": 241,
    "race_middle_Taking_a_test": 242,
    "yelp_polarity_reviews_0_2_0": 243,
    "adversarial_qa_droberta_tell_what_it_is": 244,
    "qasc_is_correct_2": 245,
}
# convert tasks_names_to_ids to tasks_ids_to_tasks_names
ids_to_tasks_names = {v: k for k, v in tasks_names_to_ids.items()}

with open("projects/wiki_experts/configs/adauni_task_dict.json", "r") as fp:
    tasks_names_to_ids_ada = json.load(fp)

ids_to_tasks_names_ada = {v: k for k, v in tasks_names_to_ids_ada.items()}


@dataclass
class ExpertInfo:
    """
    Stuff that we want to save about experts but will never be passed from command line
    """

    parent_node: str = None
    expert_name: str = None
    expert_task_name: str = None


class ExpertConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.load_in_8bit = False
        self.wandb_project = None
        self.tensorboard = False
        self.hf_token_hub = None
        self.hf_repo_id = None

        self.expert_name = None
        self.routing = "subject"
        self.mmlu_test_split = "test"
        self.load_module = None
        self.module_graph = None
        self.micro_batch_size = None
        self.validation_portion = 0.03

        self.expand_val_set_w_downstream = False

        self.eval_mmlu_callbacks_every = 0
        self.eval_test_set_callback_every = 0
        self.eval_rougeL_callback_every = 0
        self.test_sets_callbacks = []

        self.use_custom_valid_callback = False  # if True use custom callback to early top on eval loss  instead of lightning callback

        self.data_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")

        # training expert
        self.eval_mmlu_flag = False

        # training classfier routing
        self.num_labels = 246
        self.expert_model_path = None
        self.retrieval_model = "classifier"
        self.expert_library_path = None
        self.text_encoder_trained = False

        self.eval_metric = "loss"
        self.use_vllm = False

    def post_init(self):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size
