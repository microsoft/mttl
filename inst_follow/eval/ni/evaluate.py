import argparse
import json
import logging
import os
import string

import numpy as np
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class GPTTokenizer:
    def __init__(self):
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word.
        # But for the first word of a sentence, there is no space before it.
        # So, we remove all the added spaces ("Ġ").
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens


xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = ROUGEScore(rouge_keys="rouge1", tokenizer=xlingual_tokenizer)
    else:
        scorer = ROUGEScore(rouge_keys="rouge1", use_stemmer=True)
    scores = scorer(prediction, ground_truth)
    return scores["rouge1_fmeasure"].item()


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = ROUGEScore(rouge_keys="rougeL", tokenizer=xlingual_tokenizer)
    else:
        scorer = ROUGEScore(rouge_keys="rougeL", use_stemmer=True)
    scores = scorer(prediction, ground_truth)
    return scores["rougeL_fmeasure"].item()


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(
        references
    ), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    n_evaluated = 0
    for pred, gold in zip(predictions, references):
        if len(pred) == 0:
            continue  # Added: skip empty predictions, the ones that are missing
        n_evaluated += 1
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / n_evaluated  # len(references)
    rouge1 = 100.0 * rouge1 / n_evaluated  # len(references)
    rougeL = 100.0 * rougeL / n_evaluated  # len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))

    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(
            task_predictions, task_references, xlingual=xlingual
        )
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file",
        required=False,
        help="Jsonl file with each line corresponding to a prediction. "
        "Each json object should have an `id` and a `prediction` key.",
        default="/home/v-oostapenko/dev/mttl/inst_follow/eval/ni/ni_pred_alpaca_full16r_notrainonsoure_addeos[full,ptopt](7ogye4hj)_yahma_llama-7b-hfni-nshot0.jsonl",
    )
    parser.add_argument(
        "--reference_file",
        required=False,
        help="Jsonl file with each line corresponding to a reference. "
        "Each json object should have an `id` and a `references` key. "
        "`task_id`, `task_category` and `task_track` are optional, which will be used to "
        "compute the per-task performance, per-category performance and the performance for default (english) / xlingual Tracks.",
        default="/home/v-oostapenko/dev/mttl/inst_follow/eval/ni/test_references.jsonl",
    )
    parser.add_argument("--output_file", help="Jsonl file to write the results to.")
    parser.add_argument("--clean", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tasks = [
        "task1640_aqa1.0_answerable_unanswerable_question_classification",
        "task290_tellmewhy_question_answerability",
        "task304_numeric_fused_head_resolution",
        "task602_wikitext-103_answer_generation",
        "task033_winogrande_answer_generation",
        "task1161_coda19_title_generation",
        "task391_causal_relationship",
        "task242_tweetqa_classification",
        "task1195_disflqa_disfluent_to_fluent_conversion",
        "task890_gcwd_classification",
        "task620_ohsumed_medical_subject_headings_answer_generation",
        "task035_winogrande_question_modification_person",
        "task1529_scitail1.1_classification",
        "task402_grailqa_paraphrase_generation",
        "task1409_dart_text_generation",
        "task738_perspectrum_classification",
        "task1344_glue_entailment_classification",
        "task1152_bard_analogical_reasoning_causation",
        "task619_ohsumed_abstract_title_generation",
        "task892_gap_reverse_coreference_resolution",
        "task233_iirc_link_exists_classification",
        "task645_summarization",
        "task569_recipe_nlg_text_generation",
        "task1154_bard_analogical_reasoning_travel",
        "task202_mnli_contradiction_classification",
        "task891_gap_coreference_resolution",
        "task893_gap_fill_the_blank_coreference_resolution",
        "task1439_doqa_cooking_isanswerable",
        "task1159_bard_analogical_reasoning_containers",
        "task520_aquamuse_answer_given_in_passage",
        "task1407_dart_question_generation",
    ]
    eval_instances = {}
    with open(args.reference_file) as fin:
        for line in fin:
            instance = json.loads(line)
            # if track is not provided in the refernce file, we use set the track to `default` and use the default tokenizer in rouge-score.
            if "track" not in instance:
                instance["track"] = "default"
            eval_instances[instance["id"]] = instance

    all_predictions = {}
    count = 0
    with open(args.prediction_file) as fin:
        for line in fin:
            prediction = json.loads(line)
            id = prediction["id"]
            task = prediction["task_name"]
            # if task in tasks:
            prediction = prediction["prediction"]
            # if "Explanation:" in prediction:
            #     prediction = prediction.split("Explanation:")[0]
            #     count += 1
            if args.clean:
                prediction = prediction.split("\n")[0]
                # this part is for wizard
                
            all_predictions[id] = prediction.strip()
    print("how many explanation:", count)
    all_results = {}
    for track in ["default"]:
        print("Evaluating track:", track)
        instance_ids = [
            id for id, instance in eval_instances.items() if instance["track"] == track
        ]
        references = [eval_instances[id]["references"] for id in instance_ids]
        predictions = []
        instructions = []
        missing_predictions = []
        for id in instance_ids:
            if id in all_predictions:
                predictions.append(all_predictions[id])
            else:
                missing_predictions.append(id)
                predictions.append("")
        if missing_predictions:
            print(
                f"No prediction for {len(missing_predictions)} instances. Use empty string as prediction."
            )

        results = compute_metrics(
            predictions, references, xlingual=(track == "xlingual")
        )
        print("======== Overall Metrics ========")
        for metric, value in results.items():
            print(f"{metric}: {value}")
            all_results[f"{metric}_{track}_track"] = value

        if "task_category" in eval_instances[instance_ids[0]]:
            categories = [
                "_".join(eval_instances[id]["task_category"].lower().split())
                for id in instance_ids
            ]
            results_per_category = compute_grouped_metrics(
                predictions, references, categories, xlingual=(track == "xlingual")
            )
            print("======== Metrics per Category ========")
            for metric, value in results_per_category.items():
                print(f"{metric}: {value}")
                all_results[f"{metric}_{track}_track"] = value

        if "task_id" in eval_instances[instance_ids[0]]:
            tasks = [eval_instances[id]["task_id"] for id in instance_ids]
            results_per_task = compute_grouped_metrics(
                predictions, references, tasks, xlingual=(track == "xlingual")
            )
            print("======== Metrics per Task ========")
            for metric, value in results_per_task.items():
                print(f"{metric}: {value}")
                all_results[f"{metric}_{track}_track"] = value

    if args.output_file:
        with open(args.output_file, "w") as fout:
            json.dump(all_results, fout, indent=2)
