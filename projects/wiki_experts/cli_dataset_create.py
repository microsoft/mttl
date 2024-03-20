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


PHI_TEMPLATE = "Instruct: {}\nAnswer:"


def download_flan(
    hf_repo_id=None,
    cutoff=10_000,
    filter_zs=False,
    template_examples=False,
    fast_dev=False,
):
    dataset = load_dataset(
        "chiayewken/flan-v2", split="train[:1000]" if fast_dev else "train"
    )

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


@click.command()
@click.argument("task")
@click.argument("hf_target_id")
@click.option("--fast_dev", default=False, is_flag=True)
def main(task, hf_target_id, fast_dev):
    if task == "flan":
        download_flan(
            hf_target_id,
            cutoff=10_000,
            filter_zs=False,
            template_examples=False,
            fast_dev=fast_dev,
        )
        raise ValueError("Unknown task")


if __name__ == "__main__":
    main()
