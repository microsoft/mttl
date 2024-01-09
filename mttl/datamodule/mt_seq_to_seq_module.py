from functools import partial
import os
import numpy
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task, logger
from dataclasses import dataclass


def is_phi2_eval_task(task):
    eval_tasks = [
        "hellaswag_1_1_0",
        "ai2_arc_ARC_Challenge_1_0_0",
        "ai2_arc_ARC_Easy_1_0_0",
        "piqa_1_0_0",
        "winogrande_1_1_0",
        "bool_q_1_0_0",
        "openbookqa_0_1_0",
    ]
    return task in eval_tasks


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
            "task_source": "few_shot_{}".format(dataset[index]["task_source"])
            if modify_task_source
            else dataset[index]["task_source"],
            "split": dataset[index]["split"]
            if "split" in dataset.column_names
            else None,
        }

    augmented_dataset = dataset.map(map_to_few_shot, with_indices=True, num_proc=16)
    return augmented_dataset


def augment_few_shot(
    dataset, num_samples, tokenizer=None, max_input_length=None, seed=42
):
    """Augment the dataset with few-shot examples."""
    import tqdm

    augmented_dataset = []
    for source in tqdm.tqdm(dataset.unique("task_name")):
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


@dataclass
class FlatMultiTaskConfig(DatasetConfig):
    source_template: str = None
    augment_few_shot: int = 0


def apply_source_template_(source_template, example):
    example["source"] = source_template.format(example["source"])
    return example


def apply_source_template(dataset, source_template):
    if source_template is not None:
        dataset = dataset.map(
            partial(apply_source_template_, source_template),
            num_proc=os.environ.get("MTTL_NUM_PROC_DATASETS", 16),
        )
    return dataset


class FlatMultiTaskModule(DefaultDataModule):
    def setup_dataset(self):
        self.dataset = load_dataset(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        if "split" not in self.dataset.column_names["train"]:
            raise ValueError(
                "Dataset must have a 'split' column, try removing the dataset manually from the cache."
            )
        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            self.dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        train_dataset = apply_source_template(
            train_dataset, self.config.source_template
        )

        if self.config.augment_few_shot > 0:
            train_dataset_aug = augment_few_shot(
                train_dataset,
                self.config.augment_few_shot,
                tokenizer=self.tokenizer,
                max_input_length=self.config.max_input_length,
            )
            train_dataset_aug = train_dataset_aug.shuffle()
            train_dataset = train_dataset_aug.select(range(len(train_dataset)))

        self.train_dataset = train_dataset.filter(
            lambda x: x["split"] == "train",
            num_proc=n_proc,
            desc="Creating train set",
        )
        self.dev_dataset = train_dataset.filter(
            lambda x: x["split"] in ["validation", "valid"],
            num_proc=n_proc,
            desc="Creating valid set",
        )
        self.test_dataset = train_dataset.filter(
            lambda x: x["split"] == "test",
            num_proc=n_proc,
            desc="Creating test set",
        )

        if len(self.test_dataset) == 0:
            self.test_dataset = self.dev_dataset


@dataclass
class FlanConfig(DatasetConfig):
    include_template_type: str = "*"
    include_task_source: str = "P3,Flan2021,CoT"
    source_template: str = None
    remove_phi_eval_tasks: bool = False


def filter_template_type(include_template_type, example):
    return example["template_type"] in include_template_type


def filter_task_source(include_task_source, example):
    return example["task_source"] in include_task_source


class FlanModule(DefaultDataModule):
    def setup_dataset(self):
        dataset = load_dataset(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        if "split" not in dataset.column_names["train"]:
            raise ValueError(
                "Dataset must have a 'split' column, try removing the dataset manually from the cache."
            )

        if self.config.include_template_type != "*":
            dataset = dataset.filter(
                partial(
                    filter_template_type,
                    set(self.config.include_template_type.split(",")),
                ),
                num_proc=n_proc,
                desc="Filtering template types",
            )

        if self.config.include_task_source != "*":
            dataset = dataset.filter(
                partial(
                    filter_task_source, set(self.config.include_task_source.split(","))
                ),
                num_proc=n_proc,
                desc="Filtering task sources",
            )

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        train_dataset = apply_source_template(
            train_dataset, self.config.source_template
        )

        if "split" in dataset.column_names["train"]:
            self.train_dataset = train_dataset.filter(
                lambda x: x["split"] == "train",
                num_proc=n_proc,
                desc="Creating train set",
            )
            self.dev_dataset = train_dataset.filter(
                lambda x: x["split"] == "validation",
                num_proc=n_proc,
                desc="Creating valid set",
            )
            self.test_dataset = train_dataset.filter(
                lambda x: x["split"] == "test",
                num_proc=n_proc,
                desc="Creating test set",
            )
        else:
            self.train_dataset, self.dev_dataset = self.create_train_valid_split(
                train_dataset
            )
            self.test_dataset = self.dev_dataset

        if self.config.remove_phi_eval_tasks:
            assert not any(
                name in self.config.include_task_source.lower()
                for name in ["niv2", "*"]
            ), "niv2 not currently supported for phi-2 eval exclusion"

            self.train_dataset = self.train_dataset.filter(
                lambda x: not is_phi2_eval_task(x["task_name"]),
                num_proc=n_proc,
                desc="Filtering phi-2 eval tasks from training mixture.",
            )


@dataclass
class T0FlatConfig(DatasetConfig):
    use_templates_as_tasks: bool = False


class T0FlatModule(DefaultDataModule):
    def setup_dataset(self):
        dataset = load_dataset(self.config.dataset)

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name
        )

        if self.config.use_templates_as_tasks:

            def concat_templates_and_task(example):
                example["task_name"] = (
                    example["task_name"]
                    + "/"
                    + example["template_type"].strip().replace(" ", "_")
                )
                return example

            train_dataset = train_dataset.map(
                concat_templates_and_task,
                num_proc=os.environ.get("MTTL_NUM_PROC_DATASETS", 16),
            )

            self._task_names = sorted(list(set(train_dataset["task_name"])))
            self._task_to_id = {
                task_name: i for i, task_name in enumerate(self._task_names)
            }

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset
