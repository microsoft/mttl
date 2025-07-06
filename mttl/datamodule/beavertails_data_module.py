import os
from dataclasses import dataclass
from functools import partial

import numpy
from datasets import Dataset, concatenate_datasets

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.models.library.dataset_library import DatasetLibrary


def augment_few_shot_task(
    dataset,
    num_samples=None,
    few_shots=None,
    tokenizer=None,
    max_input_length=None,
    seed=42,
    modify_task_source=True,
):
    if num_samples is None and few_shots is None:
        raise ValueError("Either num_samples or few_shots must be specified.")

    len_dataset = len(dataset)
    rng = numpy.random.RandomState(seed)

    if few_shots is None:
        split = dataset["split"]
        train_indices = set(i for i in range(len_dataset) if split[i] == "train")

    def map_to_few_shot(_, index):
        if few_shots is None:
            index_range = list(train_indices - {index})
            index_chosen = rng.choice(index_range, size=num_samples, replace=False)
            index_chosen = list(map(int, index_chosen))  # datasets complains otherwise

            sources = [dataset[i]["source"] for i in index_chosen]
            targets = [dataset[i]["target"] for i in index_chosen]
        else:
            sources = few_shots["source"]
            targets = few_shots["target"]

        context = (
            "\n\n".join(
                [" ".join([source, target]) for source, target in zip(sources, targets)]
            )
            + "\n\n"
        )
        prompt = context + dataset[index]["source"]

        if tokenizer is not None and max_input_length is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            while (
                input_ids.shape[-1] > max_input_length
                and len(context.split("\n\n")) > 2
            ):
                context = "\n\n".join(context.split("\n\n")[:-2]) + "\n\n"
                prompt = context + dataset[index]["source"]
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        return {
            "source": prompt,
            "target": dataset[index]["target"],
            "task_name": dataset[index]["task_name"],
            "task_source": (
                "few_shot_{}".format(dataset[index]["task_source"])
                if modify_task_source
                else dataset[index]["task_source"]
            ),
            "split": (
                dataset[index]["split"] if "split" in dataset.column_names else None
            ),
        }

    augmented_dataset = dataset.map(map_to_few_shot, with_indices=True, num_proc=16)
    return augmented_dataset


def augment_few_shot(
    dataset, num_samples, tokenizer=None, max_input_length=None, seed=42
):
    """Augment the dataset with few-shot examples."""
    from tqdm.auto import tqdm

    augmented_dataset = []
    for source in tqdm(dataset.unique("task_name")):
        augmented_dataset.append(
            Dataset.from_list(
                augment_few_shot_task(
                    dataset.filter(lambda x: x["task_name"] == source),
                    num_samples,
                    tokenizer,
                    max_input_length,
                    seed,
                )
            )
        )
    return concatenate_datasets([dataset] + augmented_dataset)


def apply_template_(tokenizer, example):
    message = [
        {"role": "user", "content": example["source"]},
        {"role": "assistant", "content": example["target"]},
    ]

    tokenized_message = tokenizer.apply_chat_template(message, tokenize=False)
    # split the message into source and target
    example["source"] = tokenized_message.split("<|assistant|>")[0]
    example["target"] = "<|assistant|>" + tokenized_message.split("<|assistant|>")[1]

    return example


def apply_template(dataset, tokenizer):
    dataset = dataset.map(
        partial(apply_template_, tokenizer),
        num_proc=int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16)),
    )
    return dataset


@dataclass
class BeavertailsSafeDataModuleConfig(DatasetConfig):
    dataset: str = "zhan1993/BeaverTails_filtered_safe"


@DataModule.register("beavertails_safe", config_cls=BeavertailsSafeDataModuleConfig)
class BeavertailsSafeModule(DataModule):
    def setup_dataset(self):
        self.dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        self.dataset = self.dataset["30k_test"]
        self.dataset = self.dataset.rename_column("prompt", "source")
        self.dataset = self.dataset.rename_column("response", "target")

        n_proc = min(
            len(self.dataset), int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        )

        if self.tokenizer.chat_template is not None:
            self.dataset = apply_template(self.dataset, self.tokenizer)

        self.train_dataset = self.dataset
        self.dev_dataset = self.test_dataset = self.dataset


@dataclass
class BeavertailsConfig(DatasetConfig):
    source_template: str = None
    dataset: str = "zhan1993/BeaverTails_filtered_train"


@DataModule.register("beavertails", config_cls=BeavertailsConfig)
class BeavertailsModule(DataModule):
    def setup_dataset(self):
        self.dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        self.dataset = self.dataset.rename_column("prompt", "source")
        self.dataset = self.dataset.rename_column("response", "target")

        n_proc = min(
            len(self.dataset), int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        )

        if self.tokenizer.chat_template is not None:
            self.dataset = apply_template(self.dataset, self.tokenizer)

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            test_dataset,
        ) = maybe_filter_hf_dataset_by_task(
            self.dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        self.train_dataset = train_dataset
        self.dev_dataset = test_dataset
        self.test_dataset = test_dataset


if __name__ == "__main__":
    # config = BeavertailsConfig(
    #     dataset="zhan1993/BeaverTails_filtered_train",
    #     model="microsoft/Phi-3-mini-4k-instruct",
    # )
    # data_module = BeavertailsModule(config)

    config = BeavertailsSafeDataModuleConfig(
        dataset="zhan1993/BeaverTails_filtered_safe",
        model="microsoft/Phi-3-mini-4k-instruct",
    )
    data_module = BeavertailsSafeModule(config)
    data_module.setup_dataset()
    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        print(batch)
        break
