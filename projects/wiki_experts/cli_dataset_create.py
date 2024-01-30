from collections import defaultdict
from functools import partial
import itertools
import json
import click
from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
from promptsource import templates
import tqdm
from mttl.datamodule.t0_data_module import apply_template
from mttl.dataloader import t0_dataset_readers
from mttl.datamodule import t0_data_module
import os


NIV2_CATEGORY_TO_TASK_NO_MMLU = {
    "Question Generation": [
        "task001_quoref_question_generation",
        "task003_mctaco_question_generation_event_duration",
        "task006_mctaco_question_generation_transient_stationary",
        "task009_mctaco_question_generation_event_ordering",
        "task012_mctaco_question_generation_absolute_timepoint",
        "task015_mctaco_question_generation_frequency",
        "task023_cosmosqa_question_generation",
        "task026_drop_question_generation",
        "task029_winogrande_full_object",
        "task030_winogrande_full_person",
        "task031_winogrande_question_generation_object",
        "task032_winogrande_question_generation_person",
        "task040_qasc_question_generation",
        "task048_multirc_question_generation",
        "task060_ropes_question_generation",
        "task074_squad1.1_question_generation",
        "task082_babi_t1_single_supporting_fact_question_generation",
        "task120_zest_text_modification",
        "task166_clariq_sentence_generation",
        "task167_strategyqa_question_generation",
        "task182_duorc_question_generation",
        "task184_break_generate_question",
        "task191_hotpotqa_question_generation",
        "task193_duorc_question_generation",
        "task235_iirc_question_from_subtext_answer_generation",
        "task236_iirc_question_from_passage_answer_generation",
        "task240_tweetqa_question_generation",
        "task246_dream_question_generation",
        "task301_record_question_generation",
        "task311_race_question_generation",
        "task348_squad2.0_unanswerable_question_generation",
        "task381_boolq_question_generation",
        "task389_torque_generate_temporal_question",
        "task394_persianqa_question_generation",
        "task405_narrativeqa_question_generation",
        "task443_com_qa_ans_question_generation",
        "task461_qasper_question_generation",
        "task468_parsinlu_rc_question_generation",
        "task470_mrqa_question_generation",
        "task489_mwsc_question_generation",
        "task519_aquamuse_question_generation",
        "task568_circa_question_generation",
        "task581_socialiqa_question_generation",
        "task594_sciq_question_generation",
        "task599_cuad_question_generation",
        "task649_race_blank_question_generation",
        "task739_lhoestq_question_generation",
        "task757_msr_sqa_question_generation",
        "task821_protoqa_question_generation",
        "task836_viquiquad_question_generation",
        "task845_pubmedqa_question_generation",
        "task847_pubmedqa_question_generation",
        "task857_inquisitive_question_generation",
        "task859_prost_question_generation",
        "task860_prost_mcq_generation",
        "task861_prost_mcq_answers_generation",
        "task871_msmarco_question_generation",
        "task886_quail_question_generation",
        "task897_freebase_qa_topic_question_generation",
        "task901_freebase_qa_category_question_generation",
        "task917_coqa_question_generation",
        "task1325_qa_zre_question_generation_on_subject_relation",
        "task1326_qa_zre_question_generation_from_answer",
        "task1335_sqac_question_generation",
        "task1398_obqa_question_generation",
        "task1402_clue_question_generation",
        "task1437_doqa_cooking_question_generation",
        "task1440_doqa_movies_question_generation",
        "task1519_qa_srl_question_generation",
        "task1552_scitail_question_generation",
        "task1567_propara_question_generation",
        "task1569_cmrc2018_question_generation",
        "task1580_eqasc-perturbed_question_generation",
        "task1594_yahoo_answers_topics_question_generation",
        "task1602_webquestion_question_genreation",
        "task1609_xquad_en_question_generation",
        "task1611_xquad_es_question_generation",
        "task1637_doqa2.1_cooking_text_summarization",
        "task1638_doqa2.1_movies_text_summarization",
        "task1639_doqa2.1_travel_text_summarization",
        "task1657_gooaq_question_generation",
        "task1660_super_glue_question_generation",
        "task1665_trainglecopa_question_generation",
    ],
    "Question Answering": [
        "task002_quoref_answer_generation",
        "task004_mctaco_answer_generation_event_duration",
        "task007_mctaco_answer_generation_transient_stationary",
        "task010_mctaco_answer_generation_event_ordering",
        "task013_mctaco_answer_generation_absolute_timepoint",
        "task016_mctaco_answer_generation_frequency",
        "task024_cosmosqa_answer_generation",
        "task028_drop_answer_generation",
        "task041_qasc_answer_generation",
        "task047_miscellaneous_answering_science_questions",
        "task049_multirc_questions_needed_to_answer",
        "task051_multirc_correct_answer_single_sentence",
        "task054_multirc_write_correct_answer",
        "task058_multirc_question_answering",
        "task061_ropes_answer_generation",
        "task073_commonsenseqa_answer_generation",
        "task075_squad1.1_answer_generation",
        "task080_piqa_answer_generation",
        "task083_babi_t1_single_supporting_fact_answer_generation",
        "task084_babi_t1_single_supporting_fact_identify_relevant_fact",
        "task104_semeval_2019_task10_closed_vocabulary_mathematical_answer_generation",
        "task118_semeval_2019_task10_open_vocabulary_mathematical_answer_generation",
        "task119_semeval_2019_task10_geometric_mathematical_answer_generation",
        "task144_subjqa_question_answering",
        "task151_tomqa_find_location_easy_clean",
        "task152_tomqa_find_location_easy_noise",
        "task153_tomqa_find_location_hard_clean",
        "task154_tomqa_find_location_hard_noise",
        "task164_mcscript_question_answering_text",
        "task165_mcscript_question_answering_commonsense",
        "task170_hotpotqa_answer_generation",
        "task178_quartz_question_answering",
        "task194_duorc_answer_generation",
        "task225_english_language_answer_generation",
        "task228_arc_answer_generation_easy",
        "task229_arc_answer_generation_hard",
        "task230_iirc_passage_classification",
        "task231_iirc_link_classification",
        "task234_iirc_passage_line_answer_generation",
        "task237_iirc_answer_from_subtext_answer_generation",
        "task238_iirc_answer_from_passage_answer_generation",
        "task239_tweetqa_answer_generation",
        "task247_dream_answer_generation",
        "task302_record_classification",
        "task309_race_answer_generation",
        "task310_race_classification",
        "task332_tellmewhy_answer_generation",
        "task339_record_answer_generation",
        "task344_hybridqa_answer_generation",
        "task380_boolq_yes_no_question",
        "task385_socialiqa_incorrect_answer_generation",
        "task390_torque_text_span_selection",
        "task395_persianqa_answer_generation",
        "task444_com_qa_question_paraphrases_answer_generation",
        "task460_qasper_answer_generation",
        "task467_parsinlu_rc_answer_generation",
        "task469_mrqa_answer_generation",
        "task473_parsinlu_mc_classification",
        "task490_mwsc_options_generation",
        "task491_mwsc_answer_generation",
        "task580_socialiqa_answer_generation",
        "task582_naturalquestion_answer_generation",
        "task591_sciq_answer_generation",
        "task595_mocha_answer_generation",
        "task596_mocha_question_generation",
        "task597_cuad_answer_generation",
        "task598_cuad_answer_generation",
        "task615_moviesqa_answer_generation",
        "task669_ambigqa_answer_generation",
        "task740_lhoestq_answer_generation_quantity",
        "task741_lhoestq_answer_generation_place",
        "task742_lhoestq_answer_generation_frequency",
        "task745_ai2_arithmetic_questions_arithmetic",
        "task750_aqua_multiple_choice_answering",
        "task751_svamp_subtraction_question_answering",
        "task752_svamp_multiplication_question_answering",
        "task753_svamp_addition_question_answering",
        "task754_svamp_common-division_question_answering",
        "task758_msr_sqa_question_answer_generation",
        "task768_qed_text_span_selection",
        "task820_protoqa_answer_generation",
        "task835_mathdataset_answer_generation",
        "task837_viquiquad_answer_generation",
        "task849_pubmedqa_answer_generation",
        "task858_inquisitive_span_detection",
        "task861_asdiv_addsub_question_answering",
        "task862_asdiv_multidiv_question_answering",
        "task863_asdiv_multiop_question_answering",
        "task864_asdiv_singleop_question_answering",
        "task865_mawps_addsub_question_answering",
        "task866_mawps_multidiv_question_answering",
        "task867_mawps_multiop_question_answering",
        "task868_mawps_singleop_question_answering",
        "task870_msmarco_answer_generation",
        "task887_quail_answer_generation",
        "task898_freebase_qa_answer_generation",
        "task918_coqa_answer_generation",
        "task1130_xcsr_vi_commonsense_mc_classification",
        "task1131_xcsr_es_commonsense_mc_classification",
        "task1132_xcsr_ur_commonsense_mc_classification",
        "task1133_xcsr_nl_commonsense_mc_classification",
        "task1134_xcsr_hi_commonsense_mc_classification",
        "task1135_xcsr_en_commonsense_mc_classification",
        "task1136_xcsr_fr_commonsense_mc_classification",
        "task1137_xcsr_pt_commonsense_mc_classification",
        "task1138_xcsr_de_commonsense_mc_classification",
        "task1139_xcsr_ru_commonsense_mc_classification",
        "task1140_xcsr_pl_commonsense_mc_classification",
        "task1141_xcsr_zh_commonsense_mc_classification",
        "task1142_xcsr_ar_commonsense_mc_classification",
        "task1143_xcsr_it_commonsense_mc_classification",
        "task1144_xcsr_sw_commonsense_mc_classification",
        "task1145_xcsr_jap_commonsense_mc_classification",
        "task1286_openbookqa_question_answering",
        "task1293_kilt_tasks_hotpotqa_question_answering",
        "task1295_adversarial_qa_question_answering",
        "task1296_wiki_hop_question_answering",
        "task1297_qasc_question_answering",
        "task1327_qa_zre_answer_generation_from_question",
        "task1334_sqac_answer_generation",
        "task1378_quarel_correct_answer_generation",
        "task1380_quarel_correct_option_generation",
        "task1382_quarel_write_correct_answer",
        "task1399_obqa_answer_generation",
        "task1412_web_questions_question_answering",
        "task1419_mathqa_gain",
        "task1420_mathqa_general",
        "task1421_mathqa_other",
        "task1422_mathqa_physics",
        "task1423_mathqa_geometry",
        "task1424_mathqa_probability",
        "task1431_head_qa_answer_generation",
        "task1438_doqa_cooking_answer_generation",
        "task1441_doqa_movies_answer_generation",
        "task1520_qa_srl_answer_generation",
        "task1555_scitail_answer_generation",
        "task1564_triviaqa_answer_generation",
        "task1565_triviaqa_classification",
        "task1570_cmrc2018_answer_generation",
        "task1581_eqasc-perturbed_answer_generation",
        "task1601_webquestions_answer_generation",
        "task1608_xquad_en_answer_generation",
        "task1610_xquad_es_answer_generation",
        "task1625_disfl_qa_asnwer_generation",
        "task1656_gooaq_answer_generation",
        "task1661_super_glue_classification",
        "task1678_mathqa_answer_selection",
        "task1726_mathqa_correct_answer_generation",
        "task1727_wiqa_what_is_the_effect",
        "task1731_quartz_question_answering",
    ],
    "Wrong Candidate Generation": [
        "task005_mctaco_wrong_answer_generation_event_duration",
        "task008_mctaco_wrong_answer_generation_transient_stationary",
        "task011_mctaco_wrong_answer_generation_event_ordering",
        "task014_mctaco_wrong_answer_generation_absolute_timepoint",
        "task017_mctaco_wrong_answer_generation_frequency",
        "task025_cosmosqa_incorrect_answer_generation",
        "task042_qasc_incorrect_option_generation",
        "task055_multirc_write_incorrect_answer",
        "task081_piqa_wrong_answer_generation",
        "task135_winowhy_wrong_reason_generation",
        "task283_dream_incorrect_answer_generation",
        "task287_casehold_legal_incorrect_answer_generation",
        "task303_record_incorrect_answer_generation",
        "task331_gap_incorrect_answer_generation",
        "task454_swag_incorrect_answer_generation",
        "task492_mwsc_incorrect_answer_generation",
        "task592_sciq_incorrect_answer_generation",
        "task631_dbpedia_14_incorrect_answer_generation",
        "task759_msr_sqa_incorrect_answer_generation",
        "task919_coqa_incorrect_answer_generation",
        "task967_ruletaker_incorrect_fact_generation_based_on_given_paragraph",
        "task1379_quarel_incorrect_answer_generation",
        "task1381_quarel_incorrect_option_generation",
        "task1383_quarel_write_incorrect_answer",
        "task1400_obqa_incorrect_answer_generation",
        "task1558_jfleg_incorrect_answer_generation",
        "task1663_cedr_ru_incorrect_classification",
    ],
    "Question Understanding": [
        "task018_mctaco_temporal_reasoning_presence",
        "task019_mctaco_temporal_reasoning_category",
        "task027_drop_answer_type_generation",
        "task044_essential_terms_identifying_essential_words",
        "task046_miscellaneous_question_typing",
        "task227_clariq_classification",
        "task248_dream_classification",
        "task384_socialiqa_question_classification",
        "task462_qasper_classification",
        "task474_parsinlu_mc_classification",
        "task673_google_wellformed_query_classification",
        "task834_mathdataset_classification",
        "task899_freebase_qa_topic_generation",
        "task900_freebase_qa_category_classification",
        "task1289_trec_classification",
        "task1328_qa_zre_relation_generation_from_question",
    ],
    "Answerability Classification": [
        "task020_mctaco_span_based_question",
        "task050_multirc_answerability",
        "task226_english_language_answer_relevance_classification",
        "task232_iirc_link_number_classification",
        "task233_iirc_link_exists_classification",
        "task242_tweetqa_classification",
        "task290_tellmewhy_question_answerability",
        "task349_squad2.0_answerable_unanswerable_question_classification",
        "task396_persianqa_classification",
        "task520_aquamuse_answer_given_in_passage",
        "task1439_doqa_cooking_isanswerable",
        "task1442_doqa_movies_isanswerable",
        "task1624_disfl_qa_question_yesno_classification",
        "task1640_aqa1.0_answerable_unanswerable_question_classification",
    ],
    "Text Quality Evaluation": [
        "task021_mctaco_grammatical_logical",
        "task052_multirc_identify_bad_question",
        "task053_multirc_correct_bad_question",
        "task616_cola_classification",
        "task674_google_wellformed_query_sentence_generation",
        "task675_google_wellformed_query_sentence_generation",
        "task1186_nne_hrngo_classification",
        "task1283_hrngo_quality_classification",
        "task1284_hrngo_informativeness_classification",
        "task1341_msr_text_classification",
        "task1589_scifact_classification",
        "task1623_disfl_qa_disfluent_question_classification",
    ],
    "Toxic Language Detection": [
        "task022_cosmosqa_passage_inappropriate_binary",
        "task108_contextualabusedetection_classification",
        "task137_detoxifying-lms_classification_toxicity",
        "task286_olid_offense_judgment",
        "task322_jigsaw_classification_threat",
        "task323_jigsaw_classification_sexually_explicit",
        "task324_jigsaw_classification_disagree",
        "task325_jigsaw_classification_identity_attack",
        "task326_jigsaw_classification_obscene",
        "task327_jigsaw_classification_toxic",
        "task328_jigsaw_classification_insult",
        "task333_hateeval_classification_hate_en",
        "task334_hateeval_classification_hate_es",
        "task335_hateeval_classification_aggresive_en",
        "task336_hateeval_classification_aggresive_es",
        "task337_hateeval_classification_individual_en",
        "task338_hateeval_classification_individual_es",
        "task607_sbic_intentional_offense_binary_classification",
        "task608_sbic_sexual_offense_binary_classification",
        "task609_sbic_potentially_offense_binary_classification",
        "task838_cdt_classification",
        "task839_cdt_classification",
        "task904_hate_speech_offensive_classification",
        "task905_hate_speech_offensive_classification",
        "task1502_hatexplain_classification",
        "task1503_hatexplain_classification",
        "task1504_hatexplain_answer_generation",
        "task1537_tamil_offenseval_dravidian_classification",
        "task1538_malayalam_offenseval_dravidian_classification",
        "task1539_kannada_offenseval_dravidian_classification",
        "task1604_ethos_text_classification",
        "task1605_ethos_text_classification",
        "task1606_ethos_text_classification",
        "task1607_ethos_text_classification",
        "task1720_civil_comments_toxicity_classification",
        "task1721_civil_comments_obscenity_classification",
        "task1722_civil_comments_threat_classification",
        "task1723_civil_comments_sexuallyexplicit_classification",
        "task1724_civil_comments_insult_classification",
        "task1725_civil_comments_severtoxicity_classification",
    ],
    "Coreference Resolution": [
        "task033_winogrande_answer_generation",
        "task133_winowhy_reason_plausibility_detection",
        "task249_enhanced_wsc_pronoun_disambiguation",
        "task304_numeric_fused_head_resolution",
        "task329_gap_classification",
        "task330_gap_answer_generation",
        "task401_numeric_fused_head_reference",
        "task648_answer_generation",
        "task891_gap_coreference_resolution",
        "task892_gap_reverse_coreference_resolution",
        "task893_gap_fill_the_blank_coreference_resolution",
        "task1390_wscfixed_coreference",
        "task1391_winogrande_easy_answer_generation",
        "task1664_winobias_text_generation",
    ],
    "Question Rewriting": [
        "task034_winogrande_question_modification_object",
        "task035_winogrande_question_modification_person",
        "task121_zest_text_modification",
        "task402_grailqa_paraphrase_generation",
        "task442_com_qa_paraphrase_question_generation",
        "task670_ambigqa_question_generation",
        "task671_ambigqa_text_generation",
        "task1195_disflqa_disfluent_to_fluent_conversion",
        "task1345_glue_qqp_question_paraprashing",
        "task1562_zest_text_modification",
        "task1622_disfl_qa_text_modication",
    ],
    "Keyword Tagging": [
        "task036_qasc_topic_word_to_generate_related_fact",
        "task613_politifact_text_generation",
        "task620_ohsumed_medical_subject_headings_answer_generation",
        "task623_ohsumed_yes_no_answer_generation",
        "task645_summarization",
    ],
    "Sentence Composition": [
        "task037_qasc_generate_related_fact",
        "task038_qasc_combined_fact",
        "task184_snli_entailment_to_neutral_text_modification",
        "task185_snli_contradiction_to_neutral_text_modification",
        "task186_snli_contradiction_to_entailment_text_modification",
        "task187_snli_entailment_to_contradiction_text_modification",
        "task188_snli_neutral_to_entailment_text_modification",
        "task189_snli_neutral_to_contradiction_text_modification",
        "task203_mnli_sentence_generation",
        "task550_discofuse_sentence_generation",
        "task626_xlwic_sentence_based_on_given_word_sentence_generation",
        "task627_xlwic_word_with_same_meaning_sentence_generation",
        "task628_xlwic_word_with_different_meaning_sentence_generation",
        "task1364_hans_answer_generation",
        "task1368_healthfact_sentence_generation",
        "task1401_obqa_sentence_generation",
        "task1515_imppres_longtextgeneration",
        "task1530_scitail1.1_sentence_generation",
        "task1556_scitail_passage_generation",
        "task1613_sick_given_category_generate_sentence",
    ],
    "Overlap Extraction": [
        "task039_qasc_find_overlapping_words",
        "task281_points_of_correspondence",
    ],
    "Misc.": [
        "task043_essential_terms_answering_incomplete_questions",
        "task169_strategyqa_sentence_generation",
        "task183_rhyme_generation",
        "task305_jeopardy_answer_generation_normal",
        "task306_jeopardy_answer_generation_double",
        "task307_jeopardy_answer_generation_final",
        "task308_jeopardy_answer_generation_all",
        "task383_matres_classification",
        "task567_circa_text_generation",
        "task921_code_x_glue_information_retreival",
        "task922_event2mind_word_generation",
        "task924_event2mind_word_generation",
        "task1146_country_capital",
        "task1147_country_currency",
        "task1149_item_check_edible",
        "task1191_food_veg_nonveg",
        "task1192_food_flavor_profile",
        "task1193_food_course_classification",
        "task1314_country_abbreviation",
        "task1317_country_calling_code",
        "task1318_country_national_dish",
        "task1319_country_by_barcode_prefix",
        "task1320_country_domain_tld",
        "task1321_country_continent",
        "task1322_country_government_type",
        "task1332_check_leap_year",
        "task1333_check_validity_date_ddmmyyyy",
        "task1403_check_validity_date_mmddyyyy",
        "task1425_country_iso_numeric",
        "task1426_country_independence_year",
        "task1427_country_region_in_world",
        "task1428_country_surface_area",
        "task1498_24hour_to_12hour_clock",
        "task1507_boolean_temporal_reasoning",
        "task1571_cmrc2018_answer_generation_starting_index",
        "task1595_event2mind_text_generation_1",
        "task1596_event2mind_text_generation_2",
    ],
    "Paraphrasing": [
        "task045_miscellaneous_sentence_paraphrasing",
        "task132_dais_text_modification",
        "task177_para-nmt_paraphrasing",
        "task466_parsinlu_qqp_text_modification",
        "task770_pawsx_english_text_modification",
        "task771_pawsx_korean_text_modification",
        "task772_pawsx_french_text_modification",
        "task773_pawsx_spanish_text_modification",
        "task774_pawsx_german_text_modification",
        "task775_pawsx_chinese_text_modification",
        "task776_pawsx_japanese_text_modification",
        "task1614_sick_text_modify",
    ],
    "Answer Verification": [
        "task056_multirc_classify_correct_answer",
        "task057_multirc_classify_incorrect_answer",
        "task241_tweetqa_classification",
        "task579_socialiqa_classification",
        "task846_pubmedqa_classification",
        "task1294_wiki_qa_answer_verification",
        "task1392_superglue_multirc_answer_verification",
    ],
    "Story Composition": [
        "task059_ropes_story_generation",
        "task067_abductivenli_answer_generation",
        "task068_abductivenli_incorrect_answer_generation",
        "task071_abductivenli_answer_generation",
        "task072_abductivenli_answer_generation",
        "task103_facts2story_long_text_generation",
        "task269_csrg_counterfactual_story_generation",
        "task270_csrg_counterfactual_context_generation",
        "task853_hippocorpus_long_text_generation",
    ],
    "Program Execution": [
        "task062_bigbench_repeat_copy_logic",
        "task063_first_i_elements",
        "task064_all_elements_except_first_i",
        "task078_all_elements_except_last_i",
        "task079_conala_concat_strings",
        "task091_all_elements_from_index_i_to_j",
        "task093_conala_normalize_lists",
        "task094_conala_calculate_mean",
        "task095_conala_max_absolute_value",
        "task096_conala_list_index_subtraction",
        "task097_conala_remove_duplicates",
        "task098_conala_list_intersection",
        "task099_reverse_elements_between_index_i_and_j",
        "task100_concatenate_all_elements_from_index_i_to_j",
        "task101_reverse_and_concatenate_all_elements_from_index_i_to_j",
        "task113_count_frequency_of_letter",
        "task122_conala_list_index_addition",
        "task123_conala_sort_dictionary",
        "task124_conala_pair_averages",
        "task125_conala_pair_differences",
        "task157_count_vowels_and_consonants",
        "task158_count_frequency_of_words",
        "task159_check_frequency_of_words_in_sentence_pair",
        "task160_replace_letter_in_a_sentence",
        "task161_count_words_containing_letter",
        "task162_count_words_starting_with_letter",
        "task163_count_words_ending_with_letter",
        "task205_remove_even_elements",
        "task206_collatz_conjecture",
        "task207_max_element_lists",
        "task208_combinations_of_list",
        "task243_count_elements_in_set_intersection",
        "task244_count_elements_in_set_union",
        "task245_check_presence_in_set_intersection",
        "task267_concatenate_and_reverse_all_elements_from_index_i_to_j",
        "task365_synthetic_remove_vowels",
        "task366_synthetic_return_primes",
        "task367_synthetic_remove_floats",
        "task368_synthetic_even_or_odd_calculation",
        "task369_synthetic_remove_odds",
        "task370_synthetic_remove_divisible_by_3",
        "task371_synthetic_product_of_list",
        "task372_synthetic_palindrome_numbers",
        "task373_synthetic_round_tens_place",
        "task374_synthetic_pos_or_neg_calculation",
        "task376_reverse_order_of_words",
        "task377_remove_words_of_given_length",
        "task378_reverse_words_of_given_length",
        "task488_extract_all_alphabetical_elements_from_list_in_order",
        "task497_extract_all_numbers_from_list_in_order",
        "task499_extract_and_add_all_numbers_from_list",
        "task504_count_all_alphabetical_elements_in_list",
        "task505_count_all_numerical_elements_in_list",
        "task506_position_of_all_alphabetical_elements_in_list",
        "task507_position_of_all_numerical_elements_in_list",
        "task509_collate_of_all_alphabetical_and_numerical_elements_in_list_separately",
        "task523_find_if_numbers_or_alphabets_are_more_in_list",
        "task600_find_the_longest_common_substring_in_two_strings",
        "task605_find_the_longest_common_subsequence_in_two_lists",
        "task606_sum_of_all_numbers_in_list_between_positions_i_and_j",
        "task622_replace_alphabets_in_a_list_by_their_position_in_english_alphabet",
        "task636_extract_and_sort_unique_alphabets_in_a_list",
        "task637_extract_and_sort_unique_digits_in_a_list",
        "task755_find_longest_substring_and_replace_its_sorted_lowercase_version_in_both_lists",
        "task756_find_longert_substring_and_return_all_unique_alphabets_in_it",
        "task850_synthetic_longest_palindrome",
        "task851_synthetic_multiply_evens",
        "task852_synthetic_multiply_odds",
        "task1087_two_number_sum",
        "task1088_array_of_products",
        "task1089_check_monotonic_array",
        "task1148_maximum_ascii_value",
        "task1150_delete_max_min",
        "task1151_swap_max_min",
        "task1188_count_max_freq_char",
        "task1189_check_char_in_string",
        "task1190_add_integer_to_list",
        "task1194_kth_largest_element",
        "task1315_find_range_array",
        "task1316_remove_duplicates_string",
        "task1331_reverse_array",
        "task1404_date_conversion",
        "task1405_find_median",
        "task1406_kth_smallest_element",
        "task1443_string_to_number",
        "task1444_round_power_of_two",
        "task1445_closest_integers",
        "task1446_farthest_integers",
        "task1542_every_ith_element_from_starting",
        "task1551_every_ith_element_from_kth_element",
    ],
    "Coherence Classification": [
        "task065_timetravel_consistent_sentence_classification",
        "task066_timetravel_binary_consistency_classification",
        "task069_abductivenli_classification",
        "task070_abductivenli_incorrect_classification",
        "task298_storycloze_correct_end_classification",
        "task1573_samsum_classification",
    ],
    "Text to Code": [
        "task076_splash_correcting_sql_mistake",
        "task077_splash_explanation_to_sql",
        "task107_splash_question_to_sql",
        "task126_scan_structured_text_generation_command_action_all",
        "task128_scan_structured_text_generation_command_action_short",
        "task130_scan_structured_text_generation_command_action_long",
        "task210_logic2text_structured_text_generation",
        "task211_logic2text_classification",
        "task212_logic2text_classification",
        "task868_cfq_mcd1_explanation_to_sql",
        "task869_cfq_mcd1_sql_to_explanation",
        "task956_leetcode_420_strong_password_check",
    ],
    "Mathematics": [
        "task085_unnatural_addsub_arithmetic",
        "task086_translated_symbol_arithmetic",
        "task087_new_operator_addsub_arithmetic",
        "task090_equation_learner_algebra",
        "task092_check_prime_classification",
    ],
    "Spelling Error Detection": ["task088_identify_typo_verification"],
    "Grammar Error Detection": [
        "task089_swap_words_verification",
        "task1346_glue_cola_grammatical_correctness_classification",
        "task1416_youtube_caption_corrections_incorrect_grammar_classification",
    ],
    "Data to Text": [
        "task102_commongen_sentence_generation",
        "task677_ollie_sentence_answer_generation",
        "task760_msr_sqa_long_text_generation",
        "task957_e2e_nlg_text_generation_generate",
        "task1407_dart_question_generation",
        "task1409_dart_text_generation",
        "task1598_nyc_long_text_generation",
        "task1631_openpi_answer_generation",
        "task1728_web_nlg_data_to_text",
    ],
    "Text Completion": [
        "task105_story_cloze-rocstories_sentence_generation",
        "task138_detoxifying-lms_classification_fluency",
        "task139_detoxifying-lms_classification_topicality",
        "task140_detoxifying-lms_classification_style",
        "task156_codah_classification_adversarial",
        "task213_rocstories_correct_ending_classification",
        "task214_rocstories_incorrect_ending_classification",
        "task215_rocstories_incorrect_answer_generation",
        "task216_rocstories_correct_answer_generation",
        "task221_rocstories_two_choice_classification",
        "task222_rocstories_two_chioce_slotting_classification",
        "task268_casehold_legal_answer_generation",
        "task296_storycloze_correct_end_classification",
        "task297_storycloze_incorrect_end_classification",
        "task299_storycloze_sentence_generation",
        "task453_swag_answer_generation",
        "task455_swag_context_generation",
        "task961_ancora-ca-ner_text_auto_completion",
        "task963_librispeech_asr_next_word_prediction",
        "task964_librispeech_asr_text_auto_completion",
        "task1389_hellaswag_completion",
    ],
    "Ethics Classification": [
        "task106_scruples_ethical_judgment",
        "task224_scruples_anecdotes_ethical_judgment",
        "task498_scruples_anecdotes_whoiswrong_classification",
        "task502_scruples_anecdotes_whoiswrong_verification",
        "task503_scruples_anecdotes_isanswerable",
        "task508_scruples_dilemmas_more_ethical_isidentifiable",
    ],
    "Spam Classification": ["task109_smsspamcollection_spamsmsdetection"],
    "Code to Text": [
        "task110_logic2text_sentence_generation",
        "task127_scan_long_text_generation_action_command_all",
        "task129_scan_long_text_generation_action_command_short",
        "task131_scan_long_text_generation_action_command_long",
    ],
    "Text Simplification": [
        "task111_asset_sentence_simplification",
        "task112_asset_simple_sentence_identification",
        "task933_wiki_auto_style_transfer",
        "task934_turk_simplification",
    ],
    "Linguistic Probing": [
        "task114_is_the_given_word_longest",
        "task428_senteval_inversion",
        "task429_senteval_tense",
        "task430_senteval_subject_count",
        "task431_senteval_object_count",
        "task515_senteval_odd_word_out",
        "task516_senteval_conjoints_inversion",
        "task1559_blimp_binary_classification",
        "task1560_blimp_binary_classification",
    ],
    "Text Categorization": [
        "task115_help_advice_classification",
        "task143_odd-man-out_classification_generate_category",
        "task197_mnli_domain_answer_generation",
        "task198_mnli_domain_classification",
        "task204_mnli_same_genre_classification",
        "task274_overruling_legal_classification",
        "task280_stereoset_classification_stereotype_type",
        "task282_scruples_event_time",
        "task364_regard_social_impact_classification",
        "task375_classify_type_of_sentence_in_debate",
        "task379_agnews_topic_classification",
        "task495_semeval_headline_classification",
        "task496_semeval_answer_generation",
        "task501_scruples_anecdotes_post_type_verification",
        "task521_trivia_question_classification",
        "task612_yorubabbc_classification",
        "task617_amazonreview_category_text_generation",
        "task629_dbpedia_14_classification",
        "task632_dbpedia_14_classification",
        "task633_dbpedia_14_answer_generation",
        "task679_hope_edi_english_text_classification",
        "task680_hope_edi_tamil_text_classification",
        "task681_hope_edi_malayalam_text_classification",
        "task682_online_privacy_policy_text_classification",
        "task744_eurlex_classification",
        "task767_craigslist_bargains_classification",
        "task854_hippocorpus_classification",
        "task881_schema_guided_dstc8_classification",
        "task930_dailydialog_classification",
        "task1187_politifact_classification",
        "task1308_amazonreview_category_classification",
        "task1434_head_qa_classification",
        "task1488_sarcasmdetection_headline_classification",
        "task1489_sarcasmdetection_tweet_classification",
        "task1490_bengali_personal_hate_speech_binary_classification",
        "task1491_bengali_political_hate_speech_binary_classification",
        "task1492_bengali_religious_hate_speech_binary_classification",
        "task1493_bengali_geopolitical_hate_speech_binary_classification",
        "task1494_bengali_hate_speech_classification",
        "task1495_adverse_drug_event_classification",
        "task1541_agnews_classification",
        "task1588_tecla_classification",
        "task1592_yahoo_answers_topics_classfication",
        "task1593_yahoo_answers_topics_classification",
        "task1630_openpi_classification",
        "task1712_poki_classification",
    ],
    "Commonsense Classification": [
        "task116_com2sense_commonsense_reasoning",
        "task136_winowhy_knowledge_categorization",
        "task291_semeval_2020_task4_commonsense_validation",
        "task1196_atomic_classification_oeffect",
        "task1197_atomic_classification_oreact",
        "task1198_atomic_classification_owant",
        "task1199_atomic_classification_xattr",
        "task1200_atomic_classification_xeffect",
        "task1201_atomic_classification_xintent",
        "task1202_atomic_classification_xneed",
        "task1203_atomic_classification_xreact",
        "task1204_atomic_classification_hinderedby",
        "task1205_atomic_classification_isafter",
        "task1206_atomic_classification_isbefore",
        "task1207_atomic_classification_atlocation",
        "task1208_atomic_classification_xreason",
        "task1209_atomic_classification_objectuse",
        "task1210_atomic_classification_madeupof",
        "task1211_atomic_classification_hassubevent",
        "task1212_atomic_classification_hasproperty",
        "task1213_atomic_classification_desires",
        "task1214_atomic_classification_xwant",
        "task1215_atomic_classification_capableof",
        "task1216_atomic_classification_causes",
    ],
    "Translation": [
        "task117_spl_translation_en_de",
        "task171_spl_translation_en_es",
        "task172_spl_translation_en_fa",
        "task173_spl_translation_en_it",
        "task174_spl_translation_en_ja",
        "task175_spl_translation_en_pl",
        "task250_spl_translation_en_ar",
        "task251_spl_translation_en_fi",
        "task252_spl_translation_en_tr",
        "task253_spl_translation_en_zh",
        "task254_spl_translation_fi_en",
        "task255_spl_translation_it_en",
        "task256_spl_translation_de_en",
        "task257_spl_translation_ar_en",
        "task258_spl_translation_fa_en",
        "task259_spl_translation_tr_en",
        "task260_spl_translation_zh_en",
        "task261_spl_translation_es_en",
        "task262_spl_translation_ja_en",
        "task263_spl_translation_pl_en",
        "task271_europarl_translation",
        "task272_europarl_translation",
        "task312_europarl_sv_en_translation",
        "task313_europarl_en_sv_translation",
        "task424_hindienglish_corpora_hi_en_translation",
        "task425_hindienglish_corpora_en_hi_translation",
        "task432_alt_en_hi_translation",
        "task433_alt_hi_en_translation",
        "task435_alt_en_ja_translation",
        "task436_alt_ja_en_translation",
        "task438_eng_guj_parallel_corpus_en_gu_translation",
        "task439_eng_guj_parallel_corpus_gu_en_translation",
        "task446_opus_paracrawl_en_so_translation",
        "task448_opus_paracrawl_en_tl_translation",
        "task449_opus_paracrawl_ig_en_translation",
        "task450_opus_paracrawl_so_en_translation",
        "task451_opus_paracrawl_tl_en_translation",
        "task452_opus_paracrawl_en_ig_translation",
        "task530_europarl_en_es_translation",
        "task531_europarl_es_en_translation",
        "task535_alt_translation_ch_en",
        "task536_alt_translation_vi_en",
        "task537_alt_translation_th_en",
        "task538_alt_translation_bu_en",
        "task539_alt_translation_ma_en",
        "task540_alt_translation_la_en",
        "task541_alt_translation_kh_en",
        "task542_alt_translation_ja_en",
        "task543_alt_translation_bh_en",
        "task544_alt_translation_hi_en",
        "task545_alt_translation_fi_en",
        "task546_alt_translation_bg_en",
        "task547_alt_translation_entk_en",
        "task548_alt_translation_en_ch",
        "task549_alt_translation_en_vi",
        "task551_alt_translation_en_th",
        "task552_alt_translation_en_bu",
        "task553_alt_translation_en_ma",
        "task554_alt_translation_en_la",
        "task555_alt_translation_en_kh",
        "task556_alt_translation_en_ja",
        "task557_alt_translation_en_ba",
        "task558_alt_translation_en_hi",
        "task559_alt_translation_en_fi",
        "task560_alt_translation_en_entk",
        "task561_alt_translation_en_bg",
        "task601_flores_translation_sntoen",
        "task604_flores_translation_entosn",
        "task644_refresd_translation",
        "task650_opus100_ar_en_translation",
        "task651_opus100_en_ar_translation",
        "task652_parsinlu_en_fa_translation",
        "task653_parsinlu_fa_en_translation",
        "task654_bible_fa_en_translation",
        "task655_bible_en_fa_translation",
        "task656_quran_en_fa_translation",
        "task657_quran_fa_en_translation",
        "task658_tep_en_fa_translation",
        "task659_tep_fa_en_translation",
        "task660_mizan_fa_en_translation",
        "task661_mizan_en_fa_translation",
        "task662_global_voices_fa_en_translation",
        "task663_global_voices_en_fa_translation",
        "task762_emea_fr_sk_translation",
        "task763_emea_es_lt_translation",
        "task765_emea_bg_el_translation",
        "task777_pawsx_english_korean_translation",
        "task778_pawsx_english_french_translation",
        "task779_pawsx_english_spanish_translation",
        "task780_pawsx_english_german_translation",
        "task781_pawsx_english_chinese_translation",
        "task782_pawsx_english_japanese_translation",
        "task783_pawsx_korean_english_translation",
        "task784_pawsx_korean_french_translation",
        "task785_pawsx_korean_spanish_translation",
        "task786_pawsx_korean_german_translation",
        "task787_pawsx_korean_chinese_translation",
        "task788_pawsx_korean_japanese_translation",
        "task789_pawsx_french_english_translation",
        "task790_pawsx_french_korean_translation",
        "task791_pawsx_french_spanish_translation",
        "task792_pawsx_french_german_translation",
        "task793_pawsx_french_chinese_translation",
        "task794_pawsx_french_japanese_translation",
        "task795_pawsx_spanish_english_translation",
        "task796_pawsx_spanish_korean_translation",
        "task797_pawsx_spanish_french_translation",
        "task798_pawsx_spanish_german_translation",
        "task799_pawsx_spanish_chinese_translation",
        "task800_pawsx_spanish_japanese_translation",
        "task801_pawsx_german_english_translation",
        "task802_pawsx_german_korean_translation",
        "task803_pawsx_german_french_translation",
        "task804_pawsx_german_spanish_translation",
        "task805_pawsx_german_chinese_translation",
        "task806_pawsx_german_japanese_translation",
        "task807_pawsx_chinese_english_translation",
        "task808_pawsx_chinese_korean_translation",
        "task809_pawsx_chinese_french_translation",
        "task810_pawsx_chinese_spanish_translation",
        "task811_pawsx_chinese_german_translation",
        "task812_pawsx_chinese_japanese_translation",
        "task813_pawsx_japanese_english_translation",
        "task814_pawsx_japanese_korean_translation",
        "task815_pawsx_japanese_french_translation",
        "task816_pawsx_japanese_spanish_translation",
        "task817_pawsx_japanese_german_translation",
        "task818_pawsx_japanese_chinese_translation",
        "task829_giga_fren_translation",
        "task830_poleval2019_mt_translation",
        "task840_para_pdt_en_es_translation",
        "task841_para_pdt_de_en_translation",
        "task842_para_pdt_cs_en_translation",
        "task872_opus_xhosanavy_translation_eng_xhosa",
        "task873_opus_xhosanavy_translation_xhosa_eng",
        "task877_kde4_translation",
        "task878_kde4_translation",
        "task911_bianet_translation",
        "task913_bianet_translation",
        "task914_bianet_translation",
        "task977_pib_translation_oriya_urdu",
        "task978_pib_translation_urdu_oriya",
        "task979_pib_translation_malayalam_oriya",
        "task980_pib_translation_oriya_malayalam",
        "task981_pib_translation_bengali_tamil",
        "task982_pib_translation_tamil_bengali",
        "task983_pib_translation_gujarati_marathi",
        "task984_pib_translation_marathi_gujarati",
        "task985_pib_translation_hindi_oriya",
        "task986_pib_translation_oriya_hindi",
        "task987_pib_translation_english_oriya",
        "task988_pib_translation_oriya_english",
        "task989_pib_translation_marathi_urdu",
        "task990_pib_translation_urdu_marathi",
        "task991_pib_translation_english_tamil",
        "task992_pib_translation_tamil_english",
        "task993_pib_translation_hindi_tamil",
        "task994_pib_translation_tamil_hindi",
        "task995_pib_translation_bengali_english",
        "task996_pib_translation_english_bengali",
        "task997_pib_translation_bengali_oriya",
        "task998_pib_translation_oriya_bengali",
        "task999_pib_translation_malayalam_tamil",
        "task1000_pib_translation_tamil_malayalam",
        "task1001_pib_translation_gujarati_urdu",
        "task1002_pib_translation_urdu_gujarati",
        "task1003_pib_translation_bengali_malayalam",
        "task1004_pib_translation_malayalam_bengali",
        "task1005_pib_translation_malayalam_punjabi",
        "task1006_pib_translation_punjabi_malayalam",
        "task1007_pib_translation_english_punjabi",
        "task1008_pib_translation_punjabi_english",
        "task1009_pib_translation_bengali_hindi",
        "task1010_pib_translation_hindi_bengali",
        "task1011_pib_translation_hindi_punjabi",
        "task1012_pib_translation_punjabi_hindi",
        "task1013_pib_translation_gujarati_telugu",
        "task1014_pib_translation_telugu_gujarati",
        "task1015_pib_translation_punjabi_tamil",
        "task1016_pib_translation_tamil_punjabi",
        "task1017_pib_translation_hindi_malayalam",
        "task1018_pib_translation_malayalam_hindi",
        "task1019_pib_translation_oriya_telugu",
        "task1020_pib_translation_telugu_oriya",
        "task1021_pib_translation_english_malayalam",
        "task1022_pib_translation_malayalam_english",
        "task1023_pib_translation_english_hindi",
        "task1024_pib_translation_hindi_english",
        "task1025_pib_translation_bengali_punjabi",
        "task1026_pib_translation_punjabi_bengali",
        "task1027_pib_translation_marathi_telugu",
        "task1028_pib_translation_telugu_marathi",
        "task1029_pib_translation_marathi_punjabi",
        "task1030_pib_translation_punjabi_marathi",
        "task1031_pib_translation_bengali_telugu",
        "task1032_pib_translation_telugu_bengali",
        "task1033_pib_translation_gujarati_hindi",
        "task1034_pib_translation_hindi_gujarati",
        "task1035_pib_translation_tamil_urdu",
        "task1036_pib_translation_urdu_tamil",
        "task1037_pib_translation_telugu_urdu",
        "task1038_pib_translation_urdu_telugu",
        "task1039_pib_translation_oriya_punjabi",
        "task1040_pib_translation_punjabi_oriya",
        "task1041_pib_translation_gujarati_malayalam",
        "task1042_pib_translation_malayalam_gujarati",
        "task1043_pib_translation_gujarati_punjabi",
        "task1044_pib_translation_punjabi_gujarati",
        "task1045_pib_translation_hindi_telugu",
        "task1046_pib_translation_telugu_hindi",
        "task1047_pib_translation_english_telugu",
        "task1048_pib_translation_telugu_english",
        "task1049_pib_translation_malayalam_telugu",
        "task1050_pib_translation_telugu_malayalam",
        "task1051_pib_translation_punjabi_urdu",
        "task1052_pib_translation_urdu_punjabi",
        "task1053_pib_translation_hindi_urdu",
        "task1054_pib_translation_urdu_hindi",
        "task1055_pib_translation_marathi_oriya",
        "task1056_pib_translation_oriya_marathi",
        "task1057_pib_translation_english_urdu",
        "task1058_pib_translation_urdu_english",
        "task1059_pib_translation_malayalam_urdu",
        "task1060_pib_translation_urdu_malayalam",
        "task1061_pib_translation_bengali_marathi",
        "task1062_pib_translation_marathi_bengali",
        "task1063_pib_translation_gujarati_tamil",
        "task1064_pib_translation_tamil_gujarati",
        "task1065_pib_translation_punjabi_telugu",
        "task1066_pib_translation_telugu_punjabi",
        "task1067_pib_translation_bengali_gujarati",
        "task1068_pib_translation_gujarati_bengali",
        "task1069_pib_translation_bengali_urdu",
        "task1070_pib_translation_urdu_bengali",
        "task1071_pib_translation_malayalam_marathi",
        "task1072_pib_translation_marathi_malayalam",
        "task1073_pib_translation_oriya_tamil",
        "task1074_pib_translation_tamil_oriya",
        "task1075_pib_translation_tamil_telugu",
        "task1076_pib_translation_telugu_tamil",
        "task1077_pib_translation_gujarati_oriya",
        "task1078_pib_translation_oriya_gujarati",
        "task1079_pib_translation_english_gujarati",
        "task1080_pib_translation_gujarati_english",
        "task1081_pib_translation_hindi_marathi",
        "task1082_pib_translation_marathi_hindi",
        "task1083_pib_translation_marathi_tamil",
        "task1084_pib_translation_tamil_marathi",
        "task1085_pib_translation_english_marathi",
        "task1086_pib_translation_marathi_english",
        "task1090_ted_translation_en_gl",
        "task1091_ted_translation_en_it",
        "task1092_ted_translation_en_pl",
        "task1093_ted_translation_en_fa",
        "task1094_ted_translation_en_pt",
        "task1095_ted_translation_ja_gl",
        "task1096_ted_translation_ja_it",
        "task1097_ted_translation_ja_pl",
        "task1098_ted_translation_ja_fa",
        "task1099_ted_translation_ja_pt",
        "task1100_ted_translation_es_gl",
        "task1101_ted_translation_es_it",
        "task1102_ted_translation_es_pl",
        "task1103_ted_translation_es_fa",
        "task1104_ted_translation_es_pt",
        "task1105_ted_translation_ar_gl",
        "task1106_ted_translation_ar_it",
        "task1107_ted_translation_ar_pl",
        "task1108_ted_translation_ar_fa",
        "task1109_ted_translation_ar_pt",
        "task1110_ted_translation_he_gl",
        "task1111_ted_translation_he_it",
        "task1112_ted_translation_he_pl",
        "task1113_ted_translation_he_fa",
        "task1114_ted_translation_he_pt",
        "task1115_alt_ja_id_translation",
        "task1116_alt_id_ja_translation",
        "task1118_alt_ja_fil_translation",
        "task1119_alt_fil_ja_translation",
        "task1121_alt_ja_khm_translation",
        "task1122_alt_khm_ja_translation",
        "task1124_alt_ja_lo_translation",
        "task1125_alt_lo_ja_translation",
        "task1127_alt_ja_th_translation",
        "task1128_alt_th_ja_translation",
        "task1218_ted_translation_en_ja",
        "task1219_ted_translation_en_es",
        "task1220_ted_translation_en_ar",
        "task1221_ted_translation_en_he",
        "task1222_ted_translation_ja_en",
        "task1223_ted_translation_ja_es",
        "task1224_ted_translation_ja_ar",
        "task1225_ted_translation_ja_he",
        "task1226_ted_translation_es_en",
        "task1227_ted_translation_es_ja",
        "task1228_ted_translation_es_ar",
        "task1229_ted_translation_es_he",
        "task1230_ted_translation_ar_en",
        "task1231_ted_translation_ar_ja",
        "task1232_ted_translation_ar_es",
        "task1233_ted_translation_ar_he",
        "task1234_ted_translation_he_en",
        "task1235_ted_translation_he_ja",
        "task1236_ted_translation_he_es",
        "task1237_ted_translation_he_ar",
        "task1238_ted_translation_gl_en",
        "task1239_ted_translation_gl_ja",
        "task1240_ted_translation_gl_es",
        "task1241_ted_translation_gl_ar",
        "task1242_ted_translation_gl_he",
        "task1243_ted_translation_gl_it",
        "task1244_ted_translation_gl_pl",
        "task1245_ted_translation_gl_fa",
        "task1246_ted_translation_gl_pt",
        "task1247_ted_translation_it_en",
        "task1248_ted_translation_it_ja",
        "task1249_ted_translation_it_es",
        "task1250_ted_translation_it_ar",
        "task1251_ted_translation_it_he",
        "task1252_ted_translation_it_gl",
        "task1253_ted_translation_it_pl",
        "task1254_ted_translation_it_fa",
        "task1255_ted_translation_it_pt",
        "task1256_ted_translation_pl_en",
        "task1257_ted_translation_pl_ja",
        "task1258_ted_translation_pl_es",
        "task1259_ted_translation_pl_ar",
        "task1260_ted_translation_pl_he",
        "task1261_ted_translation_pl_gl",
        "task1262_ted_translation_pl_it",
        "task1263_ted_translation_pl_fa",
        "task1264_ted_translation_pl_pt",
        "task1265_ted_translation_fa_en",
        "task1266_ted_translation_fa_ja",
        "task1267_ted_translation_fa_es",
        "task1268_ted_translation_fa_ar",
        "task1269_ted_translation_fa_he",
        "task1270_ted_translation_fa_gl",
        "task1271_ted_translation_fa_it",
        "task1272_ted_translation_fa_pl",
        "task1273_ted_translation_fa_pt",
        "task1274_ted_translation_pt_en",
        "task1275_ted_translation_pt_ja",
        "task1276_ted_translation_pt_es",
        "task1277_ted_translation_pt_ar",
        "task1278_ted_translation_pt_he",
        "task1279_ted_translation_pt_gl",
        "task1280_ted_translation_pt_it",
        "task1281_ted_translation_pt_pl",
        "task1282_ted_translation_pt_fa",
        "task1323_open_subtitles_hi_en_translation",
        "task1324_open_subtitles_te_en_translation",
        "task1329_open_subtitles_en_hi_translation",
        "task1330_open_subtitles_en_te_translation",
        "task1350_opus100_translation_en_gu",
        "task1351_opus100_translation_gu_en",
        "task1352_hind_encorp_translation_hi_en",
        "task1353_hind_encorp_translation_en_hi",
        "task1365_opustedtalks_translation",
        "task1367_opustedtalks_translation",
        "task1371_newscomm_translation",
        "task1373_newscomm_translation",
        "task1374_newscomm_translation",
        "task1375_newscomm_translation",
        "task1376_newscomm_translation",
        "task1377_newscomm_translation",
        "task1395_europa_ecdc_tm_en_sv_translation",
        "task1396_europa_ecdc_tm_en_de_translation",
        "task1397_europa_ecdc_tm_fr_en_translation",
        "task1432_head_qa_language_translation_en_to_es",
        "task1433_head_qa_language_translation_es_to_en",
        "task1435_ro_sts_parallel_language_translation_ro_to_en",
        "task1436_ro_sts_parallel_language_translation_en_to_ro",
        "task1514_flores_translation_entone",
        "task1616_cc_alligned_translate_eng_tel",
        "task1617_cc_alligned_translate_tel_eng",
        "task1619_menyo20k-mt_en_yo_translation",
        "task1620_menyo20k-mt_yo_en_translation",
        "task1647_opus_books_en-pt_translation",
        "task1648_opus_books_en-sv_translation",
        "task1649_opus_books_en-no_translation",
        "task1650_opus_books_en-fi_translation",
        "task1651_opus_books_en-es__translation",
        "task1652_opus_books_ca-en_translation",
        "task1654_mkb_translation",
        "task1655_mkb_translation",
        "task1676_xquad-ca_translation",
        "task1677_xquad-ca_translation",
        "task1685_menyo20k_translation",
        "task1686_menyo20k_translation",
        "task1689_qed_amara_translation",
        "task1690_qed_amara_translation",
        "task1691_qed_amara_translation",
        "task1692_qed_amara_translation",
    ],
    "Explanation": [
        "task134_winowhy_reason_generation",
        "task192_hotpotqa_sentence_generation",
        "task223_quartz_explanation_generation",
        "task295_semeval_2020_task4_commonsense_reasoning",
        "task593_sciq_explanation_generation",
        "task1369_healthfact_sentence_generation",
    ],
    "Word Semantics": [
        "task141_odd-man-out_classification_category",
        "task142_odd-man-out_classification_no_category",
        "task457_matres_conditional_classification",
        "task458_matres_negation_classification",
        "task459_matres_static_classification",
        "task625_xlwic_true_or_false_answer_generation",
        "task1508_wordnet_antonyms",
        "task1509_evalution_antonyms",
        "task1582_bless_hypernym_generation",
        "task1585_root09_hypernym_generation",
    ],
    "Text Matching": [
        "task145_afs_argument_similarity_death_penalty",
        "task146_afs_argument_similarity_gun_control",
        "task147_afs_argument_similarity_gay_marriage",
        "task148_afs_argument_quality_gay_marriage",
        "task149_afs_argument_quality_death_penalty",
        "task150_afs_argument_quality_gun_control",
        "task273_europarl_classification",
        "task276_enhanced_wsc_classification",
        "task289_gigaword_summarization",
        "task314_europarl_sv-en_classification",
        "task315_europarl_sv-en_language_identification",
        "task400_paws_paraphrase_classification",
        "task404_grailqa_paraphrase_validation",
        "task426_hindienglish_corpora_hi-en_classification",
        "task434_alt_en_hi_answer_generation",
        "task437_alt_en_ja_answer_generation",
        "task440_eng_guj_parallel_corpus_gu-en_classification",
        "task465_parsinlu_qqp_classification",
        "task514_argument_consequence_classification",
        "task532_europarl_en-es_classification",
        "task566_circa_classification",
        "task590_amazonfood_summary_correction_classification",
        "task624_ohsumed_question_answering",
        "task630_dbpedia_14_classification",
        "task643_refresd_classification",
        "task764_emea_bg_el_classification",
        "task831_giga_fren_classification",
        "task832_poleval2019_mt_classification",
        "task910_bianet_classification",
        "task1117_alt_ja_id_answer_generation",
        "task1120_alt_ja_fil_answer_generation",
        "task1123_alt_ja_khm_answer_generation",
        "task1126_alt_ja_lo_answer_generation",
        "task1129_alt_ja_th_answer_generation",
        "task1162_coda19_title_classification",
        "task1285_kpa_keypoint_matching",
        "task1287_glue_qqp_paraphrasing",
        "task1288_glue_mrpc_paraphrasing",
        "task1347_glue_sts-b_similarity_classification",
        "task1354_sent_comp_classification",
        "task1408_dart_similarity_classification",
        "task1587_scifact_classification",
        "task1645_medical_question_pair_dataset_text_classification",
    ],
    "Pos Tagging": [
        "task155_count_nouns_verbs",
        "task345_hybridqa_answer_generation",
        "task346_hybridqa_classification",
        "task347_hybridqa_incorrect_answer_generation",
        "task382_hybridqa_answer_generation",
        "task583_udeps_eng_coarse_pos_tagging",
        "task584_udeps_eng_fine_pos_tagging",
        "task1167_penn_treebank_coarse_pos_tagging",
        "task1168_brown_coarse_pos_tagging",
        "task1543_conll2002_parts_of_speech_tagging_answer_generation",
    ],
    "Question Decomposition": [
        "task168_strategyqa_question_decomposition",
        "task176_break_decompose_questions",
    ],
    "Information Extraction": [
        "task179_participant_extraction",
        "task180_intervention_extraction",
        "task181_outcome_extraction",
        "task292_storycommonsense_character_text_generation",
        "task388_torque_token_classification",
        "task456_matres_intention_classification",
        "task528_parsinlu_movie_aspect_detection",
        "task529_parsinlu_food_aspect_detection",
        "task578_curiosity_dialogs_answer_generation",
        "task621_ohsumed_yes_no_numerical_answer_generation",
        "task646_answer_generation",
        "task647_answer_generation",
        "task676_ollie_relationship_answer_generation",
        "task678_ollie_actual_relationship_answer_generation",
        "task683_online_privacy_policy_text_purpose_answer_generation",
        "task684_online_privacy_policy_text_information_type_generation",
        "task747_glucose_cause_emotion_detection",
        "task748_glucose_reverse_cause_event_detection",
        "task749_glucose_reverse_cause_emotion_detection",
        "task874_opus_xhosanavy_sr",
        "task926_coached_conv_pref_word_generation",
        "task958_e2e_nlg_text_generation_parse",
        "task1410_dart_relationship_extraction",
        "task1411_dart_subject_identification",
        "task1413_dart_object_identification",
        "task1451_drug_dose_extraction",
        "task1506_celebrity_minimal_dob_span",
        "task1510_evalution_relation_extraction",
        "task1517_limit_classfication",
        "task1518_limit_answer_generation",
        "task1568_propara_classification",
        "task1597_nyc_slot_filling",
        "task1666_cail2018_answer_generation",
        "task1667_cail2018_answer_generation",
    ],
    "Textual Entailment": [
        "task190_snli_classification",
        "task199_mnli_classification",
        "task200_mnli_entailment_classification",
        "task201_mnli_neutral_classification",
        "task202_mnli_contradiction_classification",
        "task463_parsinlu_entailment_classification",
        "task464_parsinlu_entailment_sentence_generation",
        "task534_farstail_entailment",
        "task640_esnli_classification",
        "task641_esnli_classification",
        "task642_esnli_classification",
        "task738_perspectrum_classification",
        "task890_gcwd_classification",
        "task935_defeasible_nli_atomic_classification",
        "task936_defeasible_nli_snli_classification",
        "task937_defeasible_nli_social_classification",
        "task970_sherliic_causal_relationship",
        "task1344_glue_entailment_classification",
        "task1385_anli_r1_entailment",
        "task1386_anli_r2_entailment",
        "task1387_anli_r3_entailment",
        "task1388_cb_entailment",
        "task1516_imppres_naturallanguageinference",
        "task1529_scitail1.1_classification",
        "task1554_scitail_classification",
        "task1612_sick_label_classification",
        "task1615_sick_tclassify_b_relation_a",
    ],
    "Sentiment Analysis": [
        "task195_sentiment140_classification",
        "task196_sentiment140_answer_generation",
        "task266_paper_reviews_reviewer_perspective_classification",
        "task284_imdb_classification",
        "task285_imdb_answer_generation",
        "task293_storycommonsense_emotion_text_generation",
        "task363_sst2_polarity_classification",
        "task397_semeval_2018_task1_tweet_anger_detection",
        "task398_semeval_2018_task1_tweet_joy_detection",
        "task399_semeval_2018_task1_tweet_sadness_detection",
        "task420_persent_document_sentiment_classification",
        "task421_persent_sentence_sentiment_classification",
        "task422_persent_sentence_sentiment_verification",
        "task423_persent_document_sentiment_verification",
        "task475_yelp_polarity_classification",
        "task476_cls_english_books_classification",
        "task477_cls_english_dvd_classification",
        "task478_cls_english_music_classification",
        "task479_cls_german_books_classification",
        "task480_cls_german_dvd_classification",
        "task481_cls_german_music_classification",
        "task482_cls_french_books_classification",
        "task483_cls_french_dvd_classification",
        "task484_cls_french_music_classification",
        "task485_cls_japanese_books_classification",
        "task486_cls_japanese_dvd_classification",
        "task487_cls_japanese_music_classification",
        "task493_review_polarity_classification",
        "task494_review_polarity_answer_generation",
        "task512_twitter_emotion_classification",
        "task517_emo_classify_emotion_of_dialogue",
        "task518_emo_different_dialogue_emotions",
        "task524_parsinlu_food_aspect_classification",
        "task525_parsinlu_movie_aspect_classification",
        "task526_parsinlu_movie_overal_classification",
        "task527_parsinlu_food_overal_classification",
        "task586_amazonfood_polarity_classification",
        "task587_amazonfood_polarity_correction_classification",
        "task588_amazonfood_rating_classification",
        "task634_allegro_reviews_classification",
        "task635_allegro_reviews_answer_generation",
        "task746_yelp_restaurant_review_classification",
        "task761_app_review_classification",
        "task819_pec_sentiment_classification",
        "task823_peixian-rtgender_sentiment_analysis",
        "task833_poem_sentiment_classification",
        "task843_financial_phrasebank_classification",
        "task844_financial_phrasebank_classification",
        "task875_emotion_classification",
        "task888_reviews_classification",
        "task889_goemotions_classification",
        "task902_deceptive_opinion_spam_classification",
        "task903_deceptive_opinion_spam_classification",
        "task923_event2mind_classifier",
        "task929_products_reviews_classification",
        "task931_dailydialog_classification",
        "task974_prachathai67k_sentiment_classification",
        "task975_prachathai67k_same_genre_classification",
        "task1292_yelp_review_full_text_categorization",
        "task1310_amazonreview_rating_classification",
        "task1311_amazonreview_rating_classification",
        "task1312_amazonreview_polarity_classification",
        "task1313_amazonreview_polarity_classification",
        "task1338_peixian_equity_evaluation_corpus_sentiment_classifier",
        "task1343_amazon_us_reviews_rating",
        "task1361_movierationales_classification",
        "task1414_ajgt_twitter_ar_classification",
        "task1496_bengali_reviews_sentiment_classification",
        "task1497_bengali_book_reviews_sentiment_classification",
        "task1532_daily_dialog_emotion_classification",
        "task1535_daily_dialog_uniqueness_classification",
        "task1536_daily_dialog_happiness_classification",
        "task1575_amazon_reviews_multi_sentiment_classification",
        "task1591_allocine_classification",
        "task1662_cedr_ru_classification",
    ],
    "Stance Detection": [
        "task209_stancedetection_classification",
        "task513_argument_stance_classification",
        "task1646_dataset_card_for_catalonia_independence_corpus_text_classification",
    ],
    "Sentence Ordering": [
        "task217_rocstories_ordering_answer_generation",
        "task218_rocstories_swap_order_answer_generation",
        "task300_storycloze_order_generation",
        "task1548_wiqa_binary_classification",
        "task1549_wiqa_answer_generation_missing_step",
    ],
    "Title Generation": [
        "task219_rocstories_title_answer_generation",
        "task220_rocstories_title_classification",
        "task288_gigaword_summarization",
        "task418_persent_title_generation",
        "task500_scruples_anecdotes_title_generation",
        "task510_reddit_tifu_title_summarization",
        "task569_recipe_nlg_text_generation",
        "task602_wikitext-103_answer_generation",
        "task619_ohsumed_abstract_title_generation",
        "task743_eurlex_summarization",
        "task769_qed_summarization",
        "task1161_coda19_title_generation",
        "task1342_amazon_us_reviews_title",
        "task1356_xlsum_title_generation",
        "task1358_xlsum_title_generation",
        "task1540_parsed_pdfs_summarization",
        "task1561_clickbait_new_bg_summarization",
        "task1586_scifact_title_generation",
        "task1659_title_generation",
    ],
    "Paper Review": ["task264_paper_reviews_accept_or_reject_classification"],
    "Language Identification": [
        "task265_paper_reviews_language_identification",
        "task427_hindienglish_corpora_hi-en_language_identification",
        "task441_eng_guj_parallel_corpus_gu-en_language_identification",
        "task447_opus_paracrawl_classification",
        "task533_europarl_es-en_language_identification",
        "task562_alt_language_identification",
        "task896_miam_language_classification",
        "task912_bianet_classification",
        "task976_pib_indian_language_identification",
        "task1370_newscomm_classification",
        "task1574_amazon_reviews_multi_language_identification",
        "task1576_amazon_reviews_multi_english_language_classification",
        "task1577_amazon_reviews_multi_japanese_language_classification",
        "task1618_cc_alligned_classify_tel_eng",
        "task1621_menyo20k-mt_en_yo_language_identification",
    ],
    "Sentence Perturbation": [
        "task275_enhanced_wsc_paraphrase_generation",
        "task406_mickey_fr_sentence_perturbation_generation",
        "task407_mickey_hi_sentence_perturbation_generation",
        "task408_mickey_it_sentence_perturbation_generation",
        "task409_mickey_nl_sentence_perturbation_generation",
        "task410_mickey_ru_sentence_perturbation_generation",
        "task411_mickey_vi_sentence_perturbation_generation",
        "task412_mickey_zh_sentence_perturbation_generation",
        "task413_mickey_en_sentence_perturbation_generation",
        "task414_mickey_ar_sentence_perturbation_generation",
        "task415_mickey_bg_sentence_perturbation_generation",
        "task416_mickey_de_sentence_perturbation_generation",
        "task417_mickey_es_sentence_perturbation_generation",
        "task1669_md_gender_bias_text_modification",
        "task1670_md_gender_bias_text_modification",
    ],
    "Fill in The Blank": [
        "task277_stereoset_sentence_generation_stereotype",
        "task278_stereoset_sentence_generation_antistereotype",
        "task572_recipe_nlg_text_generation",
        "task603_wikitext-103_fill_in_the_blank",
        "task672_nummersense",
        "task944_wiki_cloze_as_multiple_choice_question_answering",
        "task945_wiki_cloze_bn_multiple_choice_question_answering",
        "task946_wiki_cloze_gu_multiple_choice_question_answering",
        "task947_wiki_cloze_hi_multiple_choice_question_answering",
        "task948_wiki_cloze_kn_multiple_choice_question_answering",
        "task949_wiki_cloze_ml_multiple_choice_question_answering",
        "task950_wiki_cloze_mr_multiple_choice_question_answering",
        "task951_wiki_cloze_or_multiple_choice_question_answering",
        "task952_wiki_cloze_pa_multiple_choice_question_answering",
        "task953_wiki_cloze_ta_multiple_choice_question_answering",
        "task954_wiki_cloze_te_multiple_choice_question_answering",
        "task962_ancora-ca-ner_missing_word_prediction",
        "task965_librispeech_asr_missing_word_prediction",
        "task1217_atomic_answer_generation",
        "task1339_peixian_equity_evaluation_corpus_text_completion",
        "task1359_numer_sense_answer_generation",
        "task1360_numer_sense_multiple_choice_qa_generation",
    ],
    "Stereotype Detection": [
        "task279_stereoset_classification_stereotype",
        "task316_crows-pairs_classification_stereotype",
        "task317_crows-pairs_classification_stereotype_type",
        "task318_stereoset_classification_gender",
        "task319_stereoset_classification_profession",
        "task320_stereoset_classification_race",
        "task321_stereoset_classification_religion",
    ],
    "Intent Identification": [
        "task294_storycommonsense_motiv_text_generation",
        "task573_air_dialogue_classification",
        "task848_pubmedqa_classification",
        "task932_dailydialog_classification",
        "task1713_convai3_sentence_generation",
    ],
    "Gender Classification": [
        "task340_winomt_classification_gender_pro",
        "task341_winomt_classification_gender_anti",
        "task342_winomt_classification_profession_pro",
        "task343_winomt_classification_profession_anti",
        "task350_winomt_classification_gender_identifiability_pro",
        "task351_winomt_classification_gender_identifiability_anti",
        "task1336_peixian_equity_evaluation_corpus_gender_classifier",
    ],
    "Section Classification": [
        "task352_coda-19_classification",
        "task1163_coda19_section_classification",
        "task1164_coda19_section_correction_classification",
    ],
    "Negotiation Strategy Detection": [
        "task353_casino_classification_negotiation_elicit_pref",
        "task354_casino_classification_negotiation_no_need",
        "task355_casino_classification_negotiation_other_need",
        "task356_casino_classification_negotiation_self_need",
        "task357_casino_classification_negotiation_small_talk",
        "task358_casino_classification_negotiation_uv_part",
        "task359_casino_classification_negotiation_vouch_fair",
    ],
    "Dialogue Generation": [
        "task360_spolin_yesand_response_generation",
        "task361_spolin_yesand_prompt_response_classification",
        "task565_circa_answer_generation",
        "task574_air_dialogue_sentence_generation",
        "task576_curiosity_dialogs_answer_generation",
        "task611_mutual_multi_turn_dialogue",
        "task639_multi_woz_user_utterance_generation",
        "task1590_diplomacy_text_generation",
        "task1600_smcalflow_sentence_generation",
        "task1603_smcalflow_sentence_generation",
        "task1714_convai3_sentence_generation",
        "task1729_personachat_generate_next",
        "task1730_personachat_choose_next",
    ],
    "Dialogue Act Recognition": [
        "task362_spolin_yesand_prompt_response_sub_classification",
        "task879_schema_guided_dstc8_classification",
        "task880_schema_guided_dstc8_classification",
        "task1394_meta_woz_task_classification",
        "task1531_daily_dialog_type_classification",
        "task1533_daily_dialog_formal_classification",
        "task1534_daily_dialog_question_classification",
    ],
    "Irony Detection": [
        "task386_semeval_2018_task3_irony_detection",
        "task387_semeval_2018_task3_irony_classification",
    ],
    "Cause Effect Classification": [
        "task391_causal_relationship",
        "task392_inverse_causal_relationship",
        "task393_plausible_result_generation",
        "task614_glucose_cause_event_detection",
        "task827_copa_commonsense_reasoning",
        "task828_copa_commonsense_cause_effect",
        "task938_copa_hi_commonsense_reasoning",
        "task939_copa_hi_commonsense_cause_effect",
        "task940_copa_gu_commonsense_reasoning",
        "task941_copa_gu_commonsense_cause_effect",
        "task942_copa_mr_commonsense_reasoning",
        "task943_copa_mr_commonsense_cause_effect",
        "task968_xcopa_commonsense_reasoning_et",
        "task969_xcopa_commonsense_cause_effect_et",
        "task1168_xcopa_commonsense_reasoning_ht",
        "task1169_xcopa_commonsense_cause_effect_ht",
        "task1170_xcopa_commonsense_reasoning_id",
        "task1171_xcopa_commonsense_cause_effect_id",
        "task1172_xcopa_commonsense_reasoning_it",
        "task1173_xcopa_commonsense_cause_effect_it",
        "task1174_xcopa_commonsense_reasoning_sw",
        "task1175_xcopa_commonsense_cause_effect_sw",
        "task1176_xcopa_commonsense_reasoning_ta",
        "task1177_xcopa_commonsense_cause_effect_ta",
        "task1178_xcopa_commonsense_reasoning_th",
        "task1179_xcopa_commonsense_cause_effect_th",
        "task1180_xcopa_commonsense_reasoning_tr",
        "task1181_xcopa_commonsense_cause_effect_tr",
        "task1182_xcopa_commonsense_reasoning_vi",
        "task1183_xcopa_commonsense_cause_effect_vi",
        "task1184_xcopa_commonsense_reasoning_zh",
        "task1185_xcopa_commonsense_cause_effect_zh",
        "task1393_superglue_copa_text_completion",
        "task1626_copa_hr_question_answering",
        "task1627_copa_hr_classification",
        "task1628_copa_hr_question_answering",
        "task1629_copa_hr_classification",
    ],
    "Fact Verification": [
        "task403_creak_commonsense_inference",
        "task966_ruletaker_fact_checking_based_on_given_context",
        "task1366_healthfact_classification",
    ],
    "Named Entity Recognition": [
        "task419_persent_answer_generation",
        "task570_recipe_nlg_ner_generation",
        "task571_recipe_nlg_ner_generation",
        "task610_conllpp_ner",
        "task959_e2e_nlg_text_generation_identify",
        "task960_ancora-ca-ner_named_entity_recognition",
        "task1447_drug_extraction_ade",
        "task1448_disease_entity_extraction_ncbi_dataset",
        "task1449_disease_entity_extraction_bc5cdr_dataset",
        "task1452_location_entity_extraction_btc_corpus",
        "task1453_person_entity_extraction_btc_corpus",
        "task1479_organization_entity_extraction_btc_corpus",
        "task1480_gene_extraction_jnlpba_dataset",
        "task1481_gene_extraction_bc2gm_dataset",
        "task1482_gene_extraction_chemprot_dataset",
        "task1483_chemical_extraction_chemprot_dataset",
        "task1484_gene_extraction_linnaeus_dataset",
        "task1485_organ_extraction_anem_dataset",
        "task1486_cell_extraction_anem_dataset",
        "task1487_organism_substance_extraction_anem_dataset",
        "task1544_conll2002_named_entity_recognition_answer_generation",
        "task1545_conll2002_person_name_extraction_answer_generation",
        "task1546_conll2002_location_name_extraction_answer_generation",
        "task1566_propara_structured_text_generation",
        "task1705_ljspeech_classification",
    ],
    "Entity Generation": ["task471_haspart_answer_generation"],
    "Entity Relation Classification": ["task472_haspart_classification"],
    "Summarization": [
        "task511_reddit_tifu_long_text_summarization",
        "task522_news_editorial_summary",
        "task589_amazonfood_summary_text_generation",
        "task618_amazonreview_summary_text_generation",
        "task668_extreme_abstract_summarization",
        "task672_amazon_and_yelp_summarization_dataset_summarization",
        "task1290_xsum_summarization",
        "task1291_multi_news_summarization",
        "task1309_amazonreview_summary_classification",
        "task1355_sent_comp_summarization",
        "task1357_xlsum_summary_generation",
        "task1499_dstc3_summarization",
        "task1553_cnn_dailymail_summarization",
        "task1572_samsum_summary",
        "task1579_gigaword_incorrect_summarization",
        "task1658_billsum_summarization",
    ],
    "Discourse Connective Identification": ["task563_discofuse_answer_generation"],
    "Discourse Relation Classification": ["task564_discofuse_classification"],
    "Speaker Identification": [
        "task575_air_dialogue_classification",
        "task577_curiosity_dialogs_classification",
        "task638_multi_woz_classification",
        "task855_conv_ai_2_classification",
        "task856_conv_ai_2_classification",
        "task906_dialogre_identify_names",
        "task909_dialogre_prevalent_speakers",
        "task925_coached_conv_pref_classifier",
        "task1599_smcalflow_classification",
    ],
    "Preposition Prediction": ["task585_preposition_classification"],
    "Dialogue State Tracking": [
        "task766_craigslist_bargains_classification",
        "task1384_deal_or_no_dialog_classification",
        "task1500_dstc3_classification",
        "task1501_dstc3_answer_generation",
    ],
    "Speaker Relation Classification": [
        "task907_dialogre_identify_relationships",
        "task908_dialogre_identify_familial_relationships",
    ],
    "Style Transfer": [
        "task927_yelp_negative_to_positive_style_transfer",
        "task928_yelp_positive_to_negative_style_transfer",
    ],
    "Sentence Expansion": ["task955_wiki_auto_style_transfer"],
    "Word Analogy": [
        "task1152_bard_analogical_reasoning_causation",
        "task1153_bard_analogical_reasoning_affordance",
        "task1154_bard_analogical_reasoning_travel",
        "task1155_bard_analogical_reasoning_trash_or_treasure",
        "task1156_bard_analogical_reasoning_tools",
        "task1157_bard_analogical_reasoning_rooms_for_containers",
        "task1158_bard_analogical_reasoning_manipulating_items",
        "task1159_bard_analogical_reasoning_containers",
    ],
    "Sentence Compression": ["task1340_msr_text_compression_compression"],
    "Grammar Error Correction": [
        "task1415_youtube_caption_corrections_grammar_correction",
        "task1557_jfleg_answer_generation",
    ],
    "Word Relation Classification": [
        "task1418_bless_semantic_relation_classification",
        "task1429_evalution_semantic_relation_classification",
        "task1505_root09_semantic_relation_classification",
        "task1583_bless_meronym_classification",
        "task1584_evalution_meronym_classification",
    ],
    "Number Conversion": [
        "task1703_ljspeech_textmodification",
        "task1704_ljspeech_textmodification",
    ],
    "Punctuation Error Detection": ["task1706_ljspeech_classification"],
    "Poem Generation": ["task1711_poki_text_generation"],
}


PHI_TEMPLATE = "Instruct: {}\nAnswer:"


def select_cutoff_by_task_name(dataset, cutoff=10_000):
    indices = []
    task_name_to_id = defaultdict(list)
    for i, task_name in enumerate(dataset["task_name"]):
        task_name_to_id[task_name].append(i)
    for task_name, indices_ in task_name_to_id.items():
        np.random.shuffle(indices_)
        indices.extend(indices_[:cutoff])
    print("Cutting off dataset:", len(indices), " / ", len(dataset))
    return dataset.select(indices)


def download_t0(cutoff=-1, per_task=True):
    dataset_folder = "t0_task"
    dataset = t0_dataset_readers.T0MixtureReader(
        t0_dataset_readers.T0DatasetConfig(
            "t0", seed=42, use_t0_templates_as_tasks=False
        )
    )

    # filter some examples from the dataset
    datasets = dataset.read_orig_dataset("train")
    all_templates = templates.TemplateCollection()

    task_dict = {}
    for task_dataset in datasets:
        template = all_templates.get_dataset(
            task_dataset.dataset_name, task_dataset.subset_name
        )[task_dataset.template_name]
        task_name = task_dataset.dataset_name + (
            ("/" + task_dataset.subset_name) if task_dataset.subset_name else ""
        )
        if task_name not in task_dict:
            task_dict[task_name] = []

        def map_func(example):
            try:
                source, target = apply_template(template, example, hash_friendly=True)
            except:
                source = "<NO INPUT>"
                target = "<NO LABEL>"
            return {
                "source": source,
                "target": target,
                "task_name": task_name,
                "template_type": task_dataset.template_name,
                "task_source": "T0",
            }

        column_names = [
            column
            for column in task_dataset.column_names
            if column
            not in ["source", "target", "task_name", "template_type", "task_source"]
        ]
        map_dataset = task_dataset.map(
            map_func, remove_columns=column_names, num_proc=32
        )
        map_dataset = map_dataset.filter(
            lambda x: x["source"] != "<NO INPUT>" and x["target"] != "<NO LABEL>"
        )
        task_dict[task_name].append(map_dataset)

    if per_task:
        for task_name, task_dataset in task_dict.items():
            print("Dumping task", task_name)
            task_dataset = concatenate_datasets(task_dataset)
            # if the dataset is too large, we randomly sample 5000 examples for the training
            if cutoff > 0:
                if len(task_dataset) > cutoff:
                    task_dataset = task_dataset.shuffle()
                    task_dataset = task_dataset.select(range(cutoff))
            # save it into the task file
            task_name = task_name.replace("/", "_")
            task_dataset.to_json(os.path.join(dataset_folder, task_name + ".json"))
    else:
        assert cutoff > 0
        all_dataset = concatenate_datasets(
            list(itertools.chain(*list(task_dict.values())))
        )
        all_dataset = all_dataset.shuffle().select(range(cutoff))
        task_names = task_dict.keys()

        for task_name in task_names:
            task_dataset = all_dataset.filter(lambda x: x["task_name"] == task_name)
            task_name = task_name.replace("/", "_")
            task_dataset.to_json(os.path.join(dataset_folder, task_name + ".json"))


def download_flan(
    hf_repo_id=None, cutoff=10_000, filter_zs=False, template_examples=False
):
    dataset = load_dataset("chiayewken/flan-v2", split="train")

    # filter some examples from the dataset
    if filter_zs:
        part = dataset.filter(
            lambda example: example["task_source"] != "NIv2", num_proc=24
        )
        part1 = part.filter(
            lambda example: example["template_type"] == "zs_noopt", num_proc=24
        )
        part2 = part.filter(
            lambda example: example["template_type"] == "zs_opt"
            and example["task_source"] == "CoT",
            num_proc=24,
        )
        dataset = concatenate_datasets([part1, part2])
        print("# number of tasks:", len(set(dataset["task_name"])))

    # group the dataset using the task_name
    task_names = dataset.unique("task_name")
    print("Num Tasks: ", len(task_names))

    all_datasets = []
    for task_name in task_names:
        print("Processing task: ", task_name)

        task_dataset = dataset.filter(
            lambda x: x["task_name"] == task_name, num_proc=24
        )

        # if the dataset is too large, we randomly sample 5000 examples for the training
        task_dataset = task_dataset.shuffle(42)

        if len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        def assign_split(example, idx):
            rng = np.random.RandomState(idx)
            draw = rng.rand()
            if draw < 0.8:
                return {"split": "train"}
            elif draw < 0.9:
                return {"split": "validation"}
            else:
                return {"split": "test"}

        task_dataset = task_dataset.map(assign_split, with_indices=True)

        if template_examples:
            from mttl.datamodule.mt_seq_to_seq_module import apply_source_template
            from mttl.datamodule.mt_seq_to_seq_module import augment_few_shot_task

            task_dataset = apply_source_template(task_dataset, PHI_TEMPLATE)
            few_shot_dataset = augment_few_shot_task(
                task_dataset, 3, modify_task_source=False
            )
            task_dataset = concatenate_datasets([task_dataset, few_shot_dataset])

        # randomly cut the dataset again
        task_dataset = task_dataset.shuffle(42)

        if len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        all_datasets.append(task_dataset)

        print("Dumping task", task_name)
        print("# Train", len(task_dataset.filter(lambda x: x["split"] == "train")))
        print("# Test", len(task_dataset.filter(lambda x: x["split"] == "test")))
        print("# Valid", len(task_dataset.filter(lambda x: x["split"] == "validation")))

    all_datasets = concatenate_datasets(all_datasets)

    def clean_task(x):
        if "task_name" not in x:
            return x

        x["task_name"] = (
            x["task_name"]
            .replace(":", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        return x

    all_datasets = all_datasets.map(lambda x: clean_task(x))
    all_datasets.push_to_hub(hf_repo_id)


def create_alpaca_code(hf_repo_id):
    import guesslang

    dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style")["train"]
    guess = guesslang.Guess()

    def map_func(guess, example):
        example["task_source"] = "code_instructions_122k_alpaca_style"
        language = guess.language_name(example["output"])
        example["task_name"] = language or "unknown"
        example["source"] = example["instruction"]
        example["target"] = example["output"]
        return example

    dataset = dataset.shuffle(42).map(partial(map_func, guess), num_proc=1)
    len_dataset_ = len(dataset)
    num_train = int(len_dataset_ * 0.95)

    def assign_split(_, idx):
        if idx < num_train:
            return {"split": "train"}
        else:
            return {"split": "validation"}

    dataset = dataset.map(assign_split, with_indices=True, num_proc=16)
    allowed_languages = set(
        ["Python", "SQL", "JavaScript", "Java", "TypeScript", "HTML"]
    )
    dataset = dataset.filter(
        lambda x: len(x["source"]) <= 219
        and len(x["source"]) >= 49
        and len(x["target"]) <= 1550
        and len(x["target"]) >= 45
        and x["task_name"] in allowed_languages,
        num_proc=16,
    )

    columns = ["source", "target", "task_name", "task_source", "split"]
    to_remove = set(dataset.column_names) - set(columns)
    dataset = dataset.remove_columns(list(to_remove))
    dataset.push_to_hub(hf_repo_id)


def create_platypus_templated(hf_repo_id):
    from mttl.dataloader.platypus_dataset_reader import PlatypusDataset

    rng = np.random.RandomState(42)
    dataset = PlatypusDataset()

    examples = []
    for example in dataset:
        examples.append(
            {
                "source": example["source"],
                "target": example["target"],
                "task_name": example["data_source"],
                "task_source": "platypus",
                "split": rng.choice(["train", "validation"], p=[0.95, 0.05]),
            }
        )

    Dataset.from_list(examples).push_to_hub(hf_repo_id)


def create_platypus_templated_instruct_answer(hf_repo_id):
    from mttl.datamodule.mt_seq_to_seq_module import apply_source_template

    dataset = load_dataset("sordonia/platypus-flat")
    apply_source_template(dataset["train"], PHI_TEMPLATE).push_to_hub(hf_repo_id)


def create_mbpp(hf_repo_id):
    from mttl.datamodule.mt_seq_to_seq_module import apply_source_template

    dataset = load_dataset("mbpp")

    def add_source(x):
        x["source"] = (
            x["text"] + "\n" + "\n".join(f">>> {test}" for test in x["test_list"])
        )
        x["task_source"] = "mbpp"
        x["task_name"] = "mbpp"
        x["target"] = x["code"]
        return x

    def add_train(x):
        x["split"] = "train"
        return x

    def add_test(x):
        x["split"] = "validation"
        return x

    dataset = dataset.map(
        add_source,
        num_proc=16,
        remove_columns=[
            "test_list",
            "text",
            "code",
            "challenge_test_list",
            "test_setup_code",
        ],
    )
    train_dataset = dataset["train"].map(add_train)
    test_dataset = dataset["validation"].map(add_test)

    apply_source_template(
        concatenate_datasets([train_dataset, test_dataset]), PHI_TEMPLATE
    ).push_to_hub(hf_repo_id)


def create_ultrachat_templated_instruct_answer(hf_repo_id):
    from mttl.datamodule.mt_seq_to_seq_module import apply_source_template

    dataset = load_dataset("sordonia/ultrachat-32c-10k-flat")
    apply_source_template(dataset["train"], PHI_TEMPLATE).push_to_hub(hf_repo_id)


def create_adauni_reduced_templated(hf_repo_id):
    dataseta = load_dataset("sordonia/flan-10k-reduced-templated-ia-flat")["train"]
    datasetb = load_dataset("sordonia/platypus-templated-ia-flat")["train"]
    datasetc = load_dataset("sordonia/ultrafeedback-templated-ia-flat")["train"]
    datasets = concatenate_datasets([dataseta, datasetb, datasetc]).shuffle(42)
    datasets.push_to_hub(hf_repo_id)


def create_adauni(hf_repo_id):
    datasets = [
        "sordonia/flan-10k-flat",
        "sordonia/platypus-flat",
        "sordonia/ultrachat-32c-10k-flat",
        "sordonia/mmlu-qa-aug-10k-flat",
    ]

    adauni = []
    columns = ["source", "target", "task_name", "task_source", "split"]

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)["train"]

        to_remove = set(dataset.column_names) - set(columns)
        dataset = dataset.remove_columns(list(to_remove))

        if dataset_name == "sordonia/flan-10k-flat":
            task_to_category = {
                task: category.replace(" ", "_").replace(".", "").lower().strip()
                for category, tasks in NIV2_CATEGORY_TO_TASK_NO_MMLU.items()
                for task in tasks
            }

            def map_func(example):
                if example["task_source"] == "NIv2":
                    if example["task_name"] in task_to_category:
                        category = task_to_category[example["task_name"]]
                        example["task_source"] = example["task_name"]
                        example["task_name"] = f"niv2_{category}"
                    else:
                        example["task_source"] = example["task_name"]
                        example["task_name"] = "niv2_no_category_found"
                return example

            dataset = dataset.map(map_func, num_proc=16)

            for category, _ in NIV2_CATEGORY_TO_TASK_NO_MMLU.items():
                category = category.replace(" ", "_").replace(".", "").lower().strip()
                category_dataset = dataset.filter(
                    lambda x: x["task_name"] == f"niv2_{category}", num_proc=16
                )
                category_dataset = category_dataset.select(
                    range(min(10_000, len(category_dataset)))
                )
                print(
                    "Built NIv2 category",
                    category,
                    "with",
                    len(category_dataset),
                    "examples",
                )
                adauni.append(category_dataset)

            other_tasks = dataset.filter(
                lambda x: not x["task_name"].startswith("niv2_"), num_proc=16
            )
            adauni.append(other_tasks)
        elif dataset_name in [
            "sordonia/platypus-flat",
            "sordonia/ultrachat-32c-10k-flat",
            "sordonia/mmlu-qa-aug-10k-flat",
        ]:
            for task_name in set(dataset["task_name"]):
                task_dataset = dataset.filter(
                    lambda x: x["task_name"] == task_name, num_proc=16
                )
                task_dataset = task_dataset.select(
                    range(min(10_000, len(task_dataset)))
                )
                adauni.append(task_dataset)
        else:
            adauni.append(dataset)

    print("Pushing to hub...")
    adauni = concatenate_datasets(adauni)
    adauni.push_to_hub(hf_repo_id)


def create_ultrafeedback_binarized_cleaned_templated_instruct_answer(hf_repo_id):
    from mttl.datamodule.mt_seq_to_seq_module import apply_source_template

    dataset = load_dataset("allenai/ultrafeedback_binarized_cleaned")

    def map_func(example):
        example["task_source"] = example["source"]
        example["task_name"] = "ultrafeedback_binarized_cleaned"
        example["source"] = example["prompt"]
        example["target"] = [
            x["content"] for x in example["chosen"] if x["role"] == "assistant"
        ][0]
        example["negative_target"] = [
            x["content"] for x in example["rejected"] if x["role"] == "assistant"
        ][0]
        return example

    # we filter out the flan examples
    dataset = dataset.filter(lambda x: "flan" not in x["source"])
    dataset = dataset.map(map_func, num_proc=16)

    train_dataset = dataset["train_sft"].map(lambda x: {"split": "train"})
    val_dataset = dataset["test_sft"].map(lambda x: {"split": "validation"})
    dataset = concatenate_datasets([train_dataset, val_dataset])
    dataset = apply_source_template(dataset, PHI_TEMPLATE)

    columns = [
        "source",
        "target",
        "task_name",
        "task_source",
        "split",
        "negative_target",
    ]
    to_remove = set(dataset.column_names) - set(columns)
    dataset = dataset.remove_columns(list(to_remove))

    dataset.push_to_hub(hf_repo_id)


def augment_with_templates_and_few_shot(
    hf_repo_id,
    source_hf_dataset_id=None,
    from_dataset=None,
    n_shot=5,
    cutoff=10_000,
):
    # Augment mmlu data with some templates and few shot examples
    templates = [
        "Instruct: {}\nResponse:",
        "Question: {}\nAnswer:",
        "Q: {}\nA:",
    ]

    if source_hf_dataset_id:
        dataset = load_dataset(source_hf_dataset_id)["train"]
    elif from_dataset:
        dataset = from_dataset
    else:
        raise ValueError("Must provide either source_hf_dataset_id or from_dataset")

    # augment the dataset few-shot
    from mttl.datamodule.mt_seq_to_seq_module import (
        augment_few_shot,
        apply_source_template,
    )

    datasets = []
    for template in templates:
        dataset_i = dataset.map(
            lambda x: apply_source_template(template, x), num_proc=16
        )
        dataset_i = augment_few_shot(dataset_i, n_shot)
        datasets.append(dataset_i)

    dataset = concatenate_datasets(datasets)
    select_cutoff_by_task_name(dataset, cutoff=cutoff).push_to_hub(hf_repo_id)


def create_sni(hf_repo_id):
    """Create a 'categorized' version of the SNI training tasks by applying a cutoff of 10k examples per category."""
    import numpy as np
    from mttl.datamodule.ni_data_module import NiDataModule, NiDataConfig

    rng = np.random.default_rng(42)

    # create a dummy datamodule
    dm = NiDataModule(
        NiDataConfig(
            "sni",
            model="EleutherAI/gpt-neo-125m",
            add_task_definition=True,
            num_pos_examples=2,
            max_input_length=2048,
            sep_by_eos_token=False,
            max_num_instances_per_task=1,
        )
    )

    datasets = []
    for category, task_list in NIV2_CATEGORY_TO_TASK_NO_MMLU.items():
        category_name = category.replace(" ", "_").replace(".", "").lower()
        category_dataset = []

        for num, task_name in enumerate(task_list):
            task_name = task_name.strip()
            try:
                for setting in ["few_shot_desc", "zero_shot_desc"]:
                    dm.config = NiDataConfig(
                        "sni",
                        model="EleutherAI/gpt-neo-125m",
                        add_task_definition=True,
                        num_pos_examples=2 if setting == "few_shot_desc" else 0,
                        max_input_length=2048,
                        sep_by_eos_token=False,
                        max_num_instances_per_task=1000,
                        finetune_task_name=task_name,
                    )
                    # for each setting, we need to setup the dataset
                    dm.setup_dataset()
                    for data in dm.train_dataloader():
                        batch_size = len(data["sources_texts"])
                        for i in range(batch_size):
                            category_dataset.append(
                                {
                                    "source": data["sources_texts"][i],
                                    "target": data["labels_texts"][i],
                                    "task_name": f"niv2_{category_name}",
                                    "task_source": f"{task_name}",
                                    "split": "train"
                                    if rng.random() <= 0.95
                                    else "validation",
                                }
                            )
            except Exception as e:
                continue

            print("# examples so far:", len(category_dataset))
            print("# tasks so far:", num + 1, "/", len(task_list))

        category_dataset = Dataset.from_list(category_dataset)
        if len(category_dataset) < 1_000:
            continue
        else:
            category_dataset = category_dataset.shuffle().select(
                range(min(10_000, len(category_dataset)))
            )
            print("Category", category_name, "Num Examples", len(category_dataset))
            datasets.append(category_dataset)

    return concatenate_datasets(datasets).push_to_hub(hf_repo_id)


def create_data(dataset_folder, hf_destination, flat=True):
    import glob
    from datasets import DatasetDict
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN")
    login(token=hf_token)

    files = glob.glob(os.path.join(dataset_folder, "*.json"))
    dataset_dict = DatasetDict()

    for file in files:
        dataset = load_dataset("json", data_files=file)["train"]

        def clean_task(x):
            if "task_name" not in x:
                return x

            x["task_name"] = (
                x["task_name"]
                .replace(":", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
            )
            return x

        dataset = dataset.map(lambda x: clean_task(x))
        task_name = dataset["task_name"][0]
        print(f"Loading {task_name}")
        dataset_dict[task_name] = dataset

    if flat:
        datasets = list(dataset_dict.values())
        concatenate_datasets(datasets).push_to_hub(hf_destination, token=hf_token)
    else:
        dataset_dict.push_to_hub(hf_destination, token=hf_token)


def create_tags(repo_id):
    # for few-shot tags, we use the flan-10k-flat dataset
    tags = [
        "zero_shot",
        "few_shot",
        "long_answer",
        "short_answer",
        "option_answer",
    ]

    def add_tag(example, tag):
        example["tags"] = "|".join([example.get("tags", ""), tag])
        return example

    def assign_zero_and_few_shot(example):
        if "opt" in example["template_type"]:
            add_tag(example, "option_answer")
        if "zs" in example["template_type"]:
            add_tag(example, "zero_shot")
        if "fs" in example["template_type"]:
            add_tag(example, "few_shot")
        if "task_source" in example:
            if "few_shot_mmlu-oai" in example["task_source"]:
                add_tag(example, "few_shot")
            elif "mmlu-oai" in example["task_source"]:
                add_tag(example, "zero_shot")
        return example

    def compute_answer_length(dataset):
        full_length = []
        for target in dataset["target"]:
            full_length.append(len(target))

        median_length = np.median(full_length)
        dataset.map(
            lambda x: add_tag(x, "short_answer")
            if len(x["target"]) < median_length
            else add_tag(x, "long_answer")
        )

    dataset = load_dataset("sordonia/flan-10k-flat")["train"]
    # zero and few shot assigned
    dataset = dataset.map(assign_zero_and_few_shot, num_proc=16)


def create_mmlu_platy():
    from datasets import DatasetInfo

    dataset = load_dataset("sordonia/qa-platy_icl0_clen128_maxD-1_maxC5000_0")
    examples = []
    num_examples = 0
    for split in dataset.keys():
        for example in dataset[split]:
            num_examples += 1
            # do some basic data filtering
            if "### Instruction:" in example["instruction"]:
                continue
            if (
                "I am sorry" in example["response"]
                or "I am not sure" in example["response"]
                or "I think" in example["response"]
            ):
                continue
            # filter if response is a question
            if example["response"].strip().endswith("?"):
                continue
            if "https://" in example["response"]:
                continue
            if not example["response"].strip().endswith("."):
                example["response"] = example["response"].strip() + "."
            examples.append(
                {
                    "source": example["instruction"].strip(),
                    "target": example["response"].strip(),
                    "task_name": split,
                    "task_source": "platy-mmlu",
                    "split": np.random.choice(["train", "validation"], p=[0.95, 0.05]),
                }
            )

    print("Length of dataset:", num_examples)
    print("Length of dataset:", len(examples))

    dataset = Dataset.from_list(
        examples, info=DatasetInfo(description="icl0_clen128_maxD-1_maxC5000_0")
    )
    augment_with_templates_and_few_shot(
        "sordonia/mmlu-qa-platy-v1-flat", from_dataset=dataset
    )


def create_dolphin_coder():
    dataset = load_dataset("cognitivecomputations/dolphin-coder")
    dataset = dataset.map(
        lambda x: {
            "source": PHI_TEMPLATE.format(x["question"]),
            "target": x["response"],
            "task_name": "dolphin-coder",
            "task_source": "dolphin-coder",
            "split": "train" if np.random.rand() < 0.95 else "validation",
        },
        remove_columns=["question", "response"],
    )
    dataset.push_to_hub("sordonia/dolphin-coder-templated-ia-flat")


@click.command()
@click.argument("task")
def main(task):
    if task == "flan":
        download_flan(cutoff=10_000)
        create_data("flan_task", "sordonia/flan-10k-flat", flat=True)
    elif task == "flan_reduced_templated":
        download_flan(
            "sordonia/flan-10k-reduced-templated-ia-flat",
            cutoff=10_000,
            filter_zs=True,
            template_examples=True,
        )
    elif task == "t0":
        download_t0(cutoff=10_000)
        create_data("t0_task", "sordonia/t0-10k-flat", flat=True)
    elif task == "sni":
        create_sni("sordonia/sni-10k-flat")
    elif task == "mbpp":
        create_mbpp("sordonia/mbpp-templated-ia-flat")
    elif task == "adauni":
        create_adauni("sordonia/adauni-v3-10k-flat")
    elif task == "ultrafeedback":
        create_ultrafeedback_binarized_cleaned_templated_instruct_answer(
            "sordonia/ultrafeedback-templated-ia-flat"
        )
    elif task == "platypus-templated":
        create_platypus_templated("sordonia/platypus-templated-flat")
    elif task == "platypus-templated-instruct-answer":
        create_platypus_templated_instruct_answer("sordonia/platypus-templated-ia-flat")
    elif task == "ultrachat-templated-instruct-answer":
        create_ultrachat_templated_instruct_answer(
            "sordonia/ultrachat-templated-ia-flat"
        )
    elif task == "tags":
        create_tags("sordonia/adauni-v4-10k-flat")
    elif task == "transform-platy":
        create_mmlu_platy()
    elif task == "alpaca_code":
        create_alpaca_code("sordonia/alpaca-code-flat")
    elif task == "adauni-reduced-templated":
        create_adauni_reduced_templated("sordonia/adauni-templated-reduced-ia-flat")
    elif task == "dolphin":
        create_dolphin_coder()
    else:
        raise ValueError("Unknown task")


if __name__ == "__main__":
    main()
