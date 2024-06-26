import argparse
import json
import logging
import os
import string

import numpy as np
from torchmetrics.text.rouge import ROUGEScore
from transformers import AutoTokenizer
from mttl.utils import logger


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


def compute_metrics(predictions, references, xlingual=False, reduction="mean"):
    assert len(predictions) == len(
        references
    ), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = [], [], []
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match.append(100. * metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        ))
        rouge1.append(100. * metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        ))
        rougeL.append(100. * metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        ))

    if reduction == "mean":
        exact_match = sum(exact_match) / len(references)
        rouge1 = sum(rouge1) / len(references)
        rougeL = sum(rougeL) / len(references)

    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
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


def compute_ni_metrics(
    preds, dataset, output_dir=None, save_prefix=None, pad_token_id=None
):
    references = [e["Instance"]["output"] for e in dataset]
    tasks = [e["Task"] for e in dataset]
    categories = [e["Categories"] for e in dataset]
    categories = ["_".join(it[0].lower().split()) for it in categories]

    result = compute_metrics(predictions=preds, references=references)
    result_per_task = compute_grouped_metrics(
        predictions=preds, references=references, groups=tasks
    )
    result.update(result_per_task)

    result_per_category = compute_grouped_metrics(
        predictions=preds, references=references, groups=categories
    )
    result.update(result_per_category)

    prediction_lens = [np.count_nonzero(pred != pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    if save_prefix is not None:
        with open(
            os.path.join(output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w"
        ) as fout:
            for example, pred in zip(dataset, preds):
                fout.write(
                    json.dumps(
                        {
                            "Task": example["Task"],
                            "Definition": example["Definition"],
                            "Instance": example["Instance"],
                            "Prediction": pred,
                        }
                    )
                    + "\n"
                )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file",
        required=True,
        help="Jsonl file with each line corresponding to a prediction. "
        "Each json object should have an `id` and a `prediction` key.",
    )
    parser.add_argument(
        "--track",
        choices=["default", "xlingual"],
        default="default",
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer.",
    )
    parser.add_argument(
        "--reference_file",
        required=False,
        help="Jsonl file with each line corresponding to a reference. "
        "Each json object should have an `id` and a `references` key. "
        "`task_id`, `task_category` and `task_track` are optional, which will be used to "
        "compute the per-task performance, per-category performance and the performance for default (english) / xlingual Tracks.",
    )
    parser.add_argument(
        "--compute_per_task_metrics",
        action="store_true",
        help="Compute metrics on every evaluation task.",
    )
    parser.add_argument("--output_file", help="Jsonl file to write the results to.")

    return parser.parse_args()


def eval_instances(args):
    eval_instances = {}
    with open(args.reference_file) as fin:
        for line in fin:
            instance = json.loads(line)
            # if track is not provided in the refernce file, we use set the track to `default` and use the default tokenizer in rouge-score.
            if "track" not in instance:
                instance["track"] = "default"
            eval_instances[instance["id"]] = instance

    all_predictions = {}
    with open(args.prediction_file) as fin:
        for line in fin:
            prediction = json.loads(line)
            id = prediction["id"]
            task = prediction["task_name"]
            # if task in tasks:
            prediction = prediction["prediction"]
            if "Input:" in prediction:
                prediction = prediction.split("Input:")[0]
            all_predictions[id] = prediction.strip()

    all_results = {}
    track = args.track
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

    results = compute_metrics(predictions, references, xlingual=(track == "xlingual"))
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
    return all_results


if __name__ == "__main__":
    args = parse_args()
    all_results = eval_instances(args)
