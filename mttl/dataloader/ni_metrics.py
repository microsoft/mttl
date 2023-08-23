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
        "--predictions", required=True, help="Path to predictions file."
    )
    parser.add_argument(
        "--track",
        choices=["default", "xlingual"],
        default="default",
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer.",
    )
    parser.add_argument(
        "--compute_per_category_metrics",
        action="store_true",
        help="Compute metrics on every evaluation category.",
    )
    parser.add_argument(
        "--compute_per_task_metrics",
        action="store_true",
        help="Compute metrics on every evaluation task.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.predictions) as fin:
        examples = [json.loads(l) for l in fin]

    predictions = [e["prediction"] for e in examples]
    references = [e["Instance"]["output"] for e in examples]
    tasks = []
    for e in examples:
        if e["Task"] == "task121_atomic_question_rewriting":
            e["Task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    results = compute_metrics(
        predictions, references, xlingual=args.track == "xlingual"
    )
    print("======== Overall Metrics ========")
    print("all_rougeL", results["rougeL"])
    print("all_EM", results["exact_match"])
    print()

    category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
    ]
    category_metrics = {
        "_".join(category.lower().split()): metric
        for category, metric in category_metrics
    }

    if args.compute_per_category_metrics:
        print("======== Metrics per category ========")
        task_category = {}
        for task in set(tasks):
            with open(os.path.join("./data/tasks/", task + ".json")) as fin:
                task_data = json.load(fin)
                task_category[task] = "_".join(
                    task_data["Categories"][0].lower().split()
                )
        categories = [task_category[e["Task"]] for e in examples]
        results.update(
            compute_grouped_metrics(
                predictions, references, categories, xlingual=args.track == "xlingual"
            )
        )

        for category, metric in category_metrics.items():
            # category = "_".join(category.lower().split())
            if f"{metric}_for_{category}" in results:
                print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
        print()

    if args.compute_per_task_metrics:
        print("======== Metrics per task ========")
        results_by_task = compute_grouped_metrics(
            predictions, references, tasks, xlingual=args.track == "xlingual"
        )
        for task in sorted(list(set(tasks))):
            category = task_category[task]
            metric = category_metrics[category]
            print(task, results_by_task[f"{metric}_for_{task}"])
        print()
