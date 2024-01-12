import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from mttl.models.utils import EfficientCheckpointModule
from projects.wiki_experts.src.ranker.adapter_ranker import AdapterRanker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentenceTransformerClassifier(AdapterRanker, EfficientCheckpointModule):
    # define the classifier, the x is the input, the task_id or expert_id is the label
    def __init__(
        self,
        task_names,
        hidden_size=768,
        transformer_embed_dim=384,
        temperature=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_encoder = self.text_encoder_init(requires_grad=False)
        self.ids_to_tasks_names = task_names
        self.task_names_to_ids = {task: i for i, task in enumerate(task_names)}
        self.num_labels = len(task_names)
        self.temperature = temperature
        # mask for available tasks
        self.available_mask: torch.Tensor = torch.ones(self.num_labels)

        # linear text encoder
        self.text_projecter = nn.Linear(transformer_embed_dim, hidden_size)
        self.out_projecter = nn.Linear(hidden_size, self.num_labels)
        self.save_hyperparameters(ignore=["text_encoder"])

    def forward(self, x):
        # Encode the text input
        text_output = torch.tensor(
            self.text_encoder.encode(x, show_progress_bar=False)
        ).to(device)
        # conver the text output to hidden vector
        text_output_projecter = self.text_projecter(text_output)
        # Calculate the logits
        logits = self.out_projecter(text_output_projecter)
        return logits

    def set_available_tasks(self, available_tasks):
        """Set the available tasks for the classifier."""
        self.available_mask.fill_(0.0)

        for task in available_tasks:
            if "default" in task:
                continue
            if (
                task in self.task_names_to_ids
            ):  # sometimes we train filtering classifiers on a subset of the tasks
                self.available_mask[self.task_names_to_ids[task]] = 1.0

    def predict_task(self, query, n=1):
        raise NotImplementedError("Not implemented yet.")

    @torch.no_grad()
    def predict_batch(self, batch, n=1, uniform="random"):
        logits = self(batch["sources_texts"]).detach().cpu()

        if self.available_mask is not None:
            logits = logits + (1.0 - self.available_mask) * -100

        # safe softmax
        max_logits = torch.max(logits, dim=1, keepdim=True).values
        logits = logits - max_logits

        expert_indices = torch.topk(logits, k=n, dim=1)

        expert_prediction = [
            [self.ids_to_tasks_names[index.item()] for index in indices]
            for indices in expert_indices.indices
        ]
        expert_weights = [
            [weight.item() for weight in weights] for weights in expert_indices.values
        ]
        # increate the entropy of the weights
        expert_weights = np.array(expert_weights) / self.temperature

        if uniform == "random":
            expert_weights = np.random.uniform(0, 1, size=expert_weights.shape)
        expert_weights = np.exp(np.array(expert_weights))
        expert_weights = expert_weights / expert_weights.sum(axis=1, keepdims=True)

        # give a uniform distribution
        if uniform == "uniform":
            expert_weights = np.ones_like(expert_weights) / len(expert_weights[0])

        return expert_prediction, expert_weights.tolist()

    def text_encoder_init(self, requires_grad=False, model_name="all-MiniLM-L6-v2"):
        text_encoder = SentenceTransformer(model_name)

        # frozen the transformer parameters
        auto_model = text_encoder._first_module().auto_model
        for param in auto_model.parameters():
            param.requires_grad = requires_grad
        return text_encoder

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        sources_texts, task_names = batch["sources_texts"], batch["task_names"]
        labels = torch.tensor([self.task_names_to_ids[task] for task in task_names]).to(
            device
        )
        logits = self(sources_texts)
        loss = F.cross_entropy(logits, labels)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sources_texts, task_names = batch["sources_texts"], batch["task_names"]
        label = torch.tensor([self.task_names_to_ids[task] for task in task_names]).to(
            device
        )
        logits = self(sources_texts)
        loss = F.cross_entropy(logits, label)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )
        return loss

    def test_step(self, batch, batch_idx):
        sources_texts, task_names = batch["sources_texts"], batch["task_names"]
        label = torch.tensor([self.task_names_to_ids[task] for task in task_names]).to(
            device
        )
        logits = self(sources_texts)
        loss = F.cross_entropy(logits, label)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )

        # compute the accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == label).item() / len(label)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)
        return loss


class ClassifierSmooth(SentenceTransformerClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dataset = load_dataset("zhan1993/transfer_matrix_v3")
        self.transfer_matrix_df = dataset["train"].to_pandas()
        self.transfer_matrix_df.set_index(["expert_name", "task_eval_on"], inplace=True)
        self.task_names_to_distribution = {}
        for task_name in self.task_names_to_ids:
            # initialize the distribution
            self.task_names_to_distribution[task_name] = np.ones(
                len(self.task_names_to_ids)
            )
            for expert_name in self.task_names_to_ids:
                # we get each expert score for each task
                if (expert_name, task_name) in self.transfer_matrix_df.index:
                    loss_score = self.transfer_matrix_df.loc[
                        (expert_name, task_name), "score"
                    ]
                else:
                    loss_score = 100
                self.task_names_to_distribution[task_name][
                    self.task_names_to_ids[expert_name]
                ] = loss_score

    def get_task_names_distribution(self, task_names):
        """
        Converts a list of task names to their corresponding scores. Here
        the scores are from the transfer matrix distribution.

        Args:
            task_names (list): A list of task names.
            [batch(task_names)]
        Returns:
            batch [batch,N] N is the number of available experts.
        """
        loss_scores = []
        for task_name in task_names:
            assert task_name in self.task_names_to_ids
            loss_scores.append(self.task_names_to_distribution[task_name])
        batch_score = torch.tensor(loss_scores).to(device)
        return batch_score

    def get_expert_distribution(self, batch):
        # get the expert distribution for the batch. [batch, N]
        expert_distribution = self.get_task_names_distribution(batch["task_names"])
        return expert_distribution

    def training_step(self, batch, batch_idx):
        logits = self(batch["sources_texts"])
        scores = self.get_expert_distribution(batch)
        scores = torch.softmax(-scores / 0.1, -1)  # note that scores are loss scores
        # logits = logits / 0.1
        probs = torch.log_softmax(logits, -1)
        loss = torch.mean(-(probs * scores).sum(1))
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["sources_texts"])
        scores = self.get_expert_distribution(batch)
        scores = torch.softmax(-scores / 0.1, -1)  # note that scores are loss scores
        # logits = logits / 0.1
        probs = torch.log_softmax(logits, -1)
        loss = torch.mean(-(probs * scores).sum(1))
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )


class ClusterPredictor(SentenceTransformerClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cluster_names = {}
        self.cluster_names_to_expert_ids = {}
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc9_2e_3epoch"
        ] = "natural_questions_open_1_0_0,web_questions_whats_the_answer,web_questions_question_answer,dbpedia_14_pick_one_category_for_the_following_text,kilt_tasks_hotpotqa_combining_facts,web_questions_short_general_knowledge_q,kilt_tasks_hotpotqa_straighforward_qa,adversarial_qa_dbidaf_generate_question,adversarial_qa_droberta_based_on,web_questions_get_the_answer,kilt_tasks_hotpotqa_complex_question,web_questions_potential_correct_answer,trivia_qa_rc_1_1_0,kilt_tasks_hotpotqa_formulate,adversarial_qa_dbert_based_on,adversarial_qa_dbidaf_based_on,squad_v1_1_3_0_0"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc2_2e_3epoch"
        ] = "adversarial_qa_dbidaf_question_context_answer,super_glue_record_1_0_2,wiki_hop_original_generate_object,adversarial_qa_droberta_tell_what_it_is,dbpedia_14_given_a_choice_of_categories_,wiki_hop_original_choose_best_object_affirmative_3,quac_1_0_0,wiki_hop_original_choose_best_object_interrogative_1,wiki_hop_original_choose_best_object_affirmative_1,adversarial_qa_dbert_answer_the_following_q,wiki_hop_original_choose_best_object_interrogative_2,adversarial_qa_droberta_question_context_answer,squad_v2_0_3_0_0,wiki_hop_original_generate_subject,wiki_bio_guess_person,adversarial_qa_dbidaf_answer_the_following_q,adversarial_qa_droberta_answer_the_following_q,adversarial_qa_dbert_tell_what_it_is,race_high_Write_a_multi_choice_question_options_given_,wiki_hop_original_choose_best_object_affirmative_2,wiki_hop_original_generate_subject_and_object,drop_2_0_0,adversarial_qa_dbert_question_context_answer,adversarial_qa_dbidaf_tell_what_it_is"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc4_2e_3epoch"
        ] = "wiki_qa_found_on_google,app_reviews_categorize_rating_using_review,race_middle_Is_this_the_right_answer,super_glue_cb_1_0_2,wiki_qa_Topic_Prediction_Answer_Only,wiki_qa_Direct_Answer_to_Question,super_glue_wsc_fixed_1_0_2,cot_gsm8k_ii,unified_qa_science_inst,race_high_Is_this_the_right_answer,cot_strategyqa,cot_ecqa_ii,quarel_do_not_use,wiki_qa_exercise,wiki_qa_automatic_system,cot_creak_ii,quarel_heres_a_story,quarel_choose_between,stream_qed_ii,wiki_qa_Topic_Prediction_Question_Only,glue_qnli_2_0_0,cot_sensemaking_ii,super_glue_copa_1_0_2,social_i_qa_Generate_the_question_from_the_answer,social_i_qa_Show_choices_and_generate_index,quarel_testing_students,wiki_qa_Topic_Prediction_Question_and_Answer_Pair,wiki_qa_Decide_good_answer,wiki_qa_Jeopardy_style,wiki_qa_Generate_Question_from_Topic,definite_pronoun_resolution_1_1_0,wiqa_effect_with_label_answer,glue_wnli_2_0_0,cot_qasc,cot_strategyqa_ii,quarel_logic_test,stream_aqua_ii"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc3_2e_3epoch"
        ] = "wiqa_what_might_be_the_first_step_of_the_process,wiqa_what_is_the_final_step_of_the_following_process,wmt16_translate_ro_en_1_0_0,wiqa_what_might_be_the_last_step_of_the_process,wiki_bio_key_content,gem_common_gen_1_1_0,duorc_SelfRC_build_story_around_qa,app_reviews_generate_review,wiki_bio_what_content,wiki_bio_who,gem_e2e_nlg_1_1_0,cot_esnli_ii,wmt16_translate_tr_en_1_0_0,wiqa_what_is_the_missing_first_step,wiki_bio_comprehension,coqa_1_0_0,duorc_ParaphraseRC_build_story_around_qa,multi_news_1_0_0"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc8_2e_3epoch"
        ] = "race_middle_Read_the_article_and_answer_the_question_no_option_,race_high_Select_the_best_answer,quail_description_context_question_answer_id,quail_context_question_description_text,race_high_Read_the_article_and_answer_the_question_no_option_,race_high_Select_the_best_answer_no_instructions_,quail_context_description_question_answer_id,race_high_Taking_a_test,super_glue_multirc_1_0_2,race_middle_Select_the_best_answer,quail_context_question_description_answer_id,quail_description_context_question_answer_text,quail_context_question_answer_description_text,race_high_Select_the_best_answer_generate_span_,race_middle_Select_the_best_answer_generate_span_,quail_context_question_answer_description_id,quail_context_description_question_answer_text,quail_context_description_question_text,quail_context_question_description_answer_text,quail_description_context_question_text,race_middle_Taking_a_test,quail_no_prompt_id,quail_no_prompt_text,race_middle_Select_the_best_answer_no_instructions_"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc5_2e_3epoch"
        ] = "quoref_Context_Contains_Answer,duorc_SelfRC_generate_question_by_answer,quoref_Find_Answer,duorc_ParaphraseRC_movie_director,duorc_ParaphraseRC_answer_question,quoref_Found_Context_Online,quoref_Read_And_Extract_,duorc_ParaphraseRC_title_generation,duorc_ParaphraseRC_decide_worth_it,quoref_What_Is_The_Answer,duorc_ParaphraseRC_generate_question,quoref_Guess_Title_For_Context,quoref_Answer_Test,duorc_SelfRC_question_answering,duorc_SelfRC_title_generation,duorc_ParaphraseRC_generate_question_by_answer,duorc_ParaphraseRC_extract_answer,duorc_SelfRC_answer_question,duorc_SelfRC_decide_worth_it,duorc_ParaphraseRC_question_answering,quoref_Answer_Question_Given_Context,duorc_SelfRC_extract_answer,quoref_Guess_Answer,quoref_Answer_Friend_Question,duorc_SelfRC_movie_director,duorc_SelfRC_generate_question,quoref_Given_Context_Answer_Question"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc1_2e_3epoch"
        ] = "glue_sst2_2_0_0,adversarial_qa_droberta_generate_question,true_case,stream_qed,huggingface_xsum,cot_esnli,cot_gsm8k,trec_1_0_0,yelp_polarity_reviews_0_2_0,lambada_1_0_0,glue_cola_2_0_0,ag_news_subset_1_0_0,gem_dart_1_1_0,math_dataset_algebra__linear_1d_1_0_0,cnn_dailymail_3_4_0,wiki_hop_original_explain_relation,dbpedia_14_given_list_what_category_does_the_paragraph_belong_to,gem_wiki_lingua_english_en_1_1_0,fix_punct,imdb_reviews_plain_text_1_0_0,race_middle_Write_a_multi_choice_question_for_the_following_article,gigaword_1_2_0,dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to,gem_web_nlg_en_1_1_0,word_segment,race_high_Write_a_multi_choice_question_for_the_following_article,wmt16_translate_de_en_1_0_0,cot_ecqa,aeslc_1_0_0,dream_generate_first_utterance,wmt16_translate_fi_en_1_0_0,dream_answer_to_dialogue,para_crawl_enes,adversarial_qa_dbert_generate_question,race_middle_Write_a_multi_choice_question_options_given_,wmt14_translate_fr_en_1_0_0"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc0_2e_3epoch"
        ] = "ropes_background_new_situation_answer,ropes_prompt_bottom_no_hint,ropes_plain_background_situation,ropes_new_situation_background_answer,ropes_given_background_situation,ropes_prompt_bottom_hint_beginning,ropes_prompt_beginning,ropes_read_background_situation,ropes_plain_bottom_hint,ropes_plain_no_background,ropes_prompt_mix,ropes_background_situation_middle"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc6_2e_3epoch"
        ] = "super_glue_rte_1_0_2,cot_sensemaking,super_glue_wic_1_0_2,cos_e_v1_11_rationale,anli_r3_0_1_0,dream_generate_last_utterance,paws_wiki_1_1_0,cos_e_v1_11_generate_explanation_given_text,cot_creak,stream_aqua,snli_1_1_0,cos_e_v1_11_i_think,glue_qqp_2_0_0,cos_e_v1_11_explain_why_human,anli_r2_0_1_0,anli_r1_0_1_0,glue_stsb_2_0_0,cos_e_v1_11_aligned_with_common_sense,glue_mnli_2_0_0,social_i_qa_I_was_wondering,cosmos_qa_1_0_0,glue_mrpc_2_0_0,social_i_qa_Generate_answer"
        self.cluster_names[
            "phi2_joint_lora_embed_clustersc7_2e_3epoch"
        ] = "dream_read_the_following_conversation_and_answer_the_question,app_reviews_convert_to_star_rating,cos_e_v1_11_question_option_description_text,social_i_qa_Show_choices_and_generate_answer,quartz_answer_question_based_on,sciq_Direct_Question_Closed_Book_,qasc_qa_with_separated_facts_3,quartz_given_the_fact_answer_the_q,quartz_answer_question_below,kilt_tasks_hotpotqa_final_exam,sciq_Multiple_Choice,wiqa_does_the_supposed_perturbation_have_an_effect,cos_e_v1_11_question_description_option_text,wiki_qa_Is_This_True_,quartz_use_info_from_question_paragraph,sciq_Direct_Question,qasc_qa_with_separated_facts_2,wiqa_which_of_the_following_is_the_supposed_perturbation,app_reviews_convert_to_rating,cos_e_v1_11_question_option_description_id,wiqa_effect_with_string_answer,qasc_qa_with_separated_facts_5,dream_baseline,quartz_having_read_above_passage,cos_e_v1_11_question_description_option_id,qasc_qa_with_separated_facts_1,cos_e_v1_11_description_question_option_text,qasc_qa_with_combined_facts_1,qasc_is_correct_1,cos_e_v1_11_description_question_option_id,social_i_qa_Check_if_a_random_answer_is_valid_or_not,sciq_Multiple_Choice_Closed_Book_,quartz_use_info_from_paragraph_question,qasc_is_correct_2,qasc_qa_with_separated_facts_4,quartz_read_passage_below_choose,quartz_paragraph_question_plain_concat,sciq_Multiple_Choice_Question_First"

        for cluster_name in self.cluster_names:
            self.cluster_names_to_expert_ids[cluster_name] = [
                self.task_names_to_ids[task_name]
                for task_name in self.cluster_names[cluster_name].split(",")
            ]

        self.cluster_names_to_ids = {
            cluster_name: i for i, cluster_name in enumerate(self.cluster_names)
        }

        self.ids_to_cluster_names = {
            i: cluster_name for i, cluster_name in enumerate(self.cluster_names)
        }

    @torch.no_grad()
    def predict_batch(self, batch, n=1, uniform="uniform"):
        logits = self(batch["sources_texts"]).detach().cpu()

        # softmax
        logits = torch.softmax(logits, dim=-1)

        # get the cluster distribution
        cluster_distribution = torch.zeros(logits.shape[0], len(self.cluster_names))
        for cluster_name in self.cluster_names_to_ids:
            cluster_distribution[
                :, self.cluster_names_to_ids[cluster_name]
            ] = torch.sum(
                logits[:, self.cluster_names_to_expert_ids[cluster_name]], dim=-1
            )

        # get the topk clusters
        cluster_indices = torch.topk(cluster_distribution, k=n, dim=1)

        cluster_prediction = [
            [self.ids_to_cluster_names[index.item()] for index in indices]
            for indices in cluster_indices.indices
        ]

        cluster_weights = [
            [weight.item() for weight in weights] for weights in cluster_indices.values
        ]
        # increate the entropy of the weights
        cluster_weights = np.array(cluster_weights) / self.temperature

        # give a random distribution
        if uniform == "random":
            cluster_weights = np.random.uniform(0, 1, size=cluster_weights.shape)
        cluster_weights = np.exp(np.array(cluster_weights))
        cluster_weights = cluster_weights / cluster_weights.sum(axis=1, keepdims=True)

        # give a uniform distribution
        if uniform == "uniform":
            cluster_weights = np.ones_like(cluster_weights) / len(cluster_weights[0])

        return cluster_prediction, cluster_weights.tolist()


class T5Classifier(SentenceTransformerClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def text_encoder_init(self, requires_grad=False, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

        # Freeze all the parameters
        for param in model.parameters():
            param.requires_grad = requires_grad
        return model

    def forward(self, x):
        # Encode the text input
        input_ids = self.tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).input_ids.to(device)

        last_hidden_states = self.text_encoder.encoder(
            input_ids=input_ids
        ).last_hidden_state

        # Pooling strategy: here, we just take the representation of the first token
        pooled_output = last_hidden_states[:, 0]

        # Calculate the logits
        text_output_projecter = self.text_projecter(pooled_output)
        # Calculate the logits
        logits = self.out_projecter(text_output_projecter)
        return logits
