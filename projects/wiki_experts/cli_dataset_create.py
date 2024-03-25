from functools import partial

import click
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def download_flan(split="train", download_size=-1, cutoff=10_000):
    dataset_name = "chiayewken/flan-v2"
    if download_size <= 0:
        dataset = load_dataset(dataset_name, split=split)
    else:
        iter_ds = load_dataset(dataset_name, split=split, streaming=True).take(
            download_size
        )
        dataset = Dataset.from_generator(
            partial(gen_from_iterable_dataset, iter_ds), features=iter_ds.features
        )

    # group the dataset using the task_name
    task_names = dataset.unique("task_name")
    print("Num Tasks: ", len(task_names))

    all_datasets = []
    for task_name in task_names:
        print("Processing task: ", task_name)

        task_dataset = dataset.filter(
            lambda x: x["task_name"] == task_name, num_proc=24
        )

        # if the dataset is too large, we randomly sample "cutoff" examples for training
        task_dataset = task_dataset.shuffle(42)

        if cutoff > 0 and len(task_dataset) > cutoff:
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
        # randomly cut the dataset again
        task_dataset = task_dataset.shuffle(42)

        if cutoff and len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        all_datasets.append(task_dataset)

        print("Dumping task", task_name)
        print("# Train", len(task_dataset.filter(lambda x: x["split"] == "train")))
        print("# Test", len(task_dataset.filter(lambda x: x["split"] == "test")))
        print("# Valid", len(task_dataset.filter(lambda x: x["split"] == "validation")))

    concatenated_datasets = concatenate_datasets(all_datasets)

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

    concatenated_datasets = concatenated_datasets.map(lambda x: clean_task(x))
    return concatenated_datasets


@click.command()
@click.argument("task")
@click.option(
    "--split",
    type=str,
    default="train",
    help="Split to download",
)
@click.option(
    "--download_size",
    type=int,
    default="-1",
    help=(
        "Download size from Hugging Face (before any processing). "
        "Values <= 0 means download the whole dataset"
    ),
)
@click.option(
    "--cutoff",
    type=int,
    default=10_000,
    help="Max number of examples per task. Zero means no cutoff.",
)
@click.option(
    "--hf_target_id",
    type=str,
    default=None,
    help="Repository ID where to push the processed dataset.",
)
def main(task, split, download_size, cutoff, hf_target_id):
    if task == "flan":
        concatenated_datasets = download_flan(
            split=split,
            download_size=download_size,
            cutoff=cutoff,
        )
    else:
        raise ValueError("Unknown task")

    if hf_target_id is not None:
        concatenated_datasets.push_to_hub(hf_target_id)


if __name__ == "__main__":
    main()
