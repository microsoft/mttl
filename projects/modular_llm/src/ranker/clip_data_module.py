import os
from dataclasses import dataclass

from mttl.datamodule.base import DefaultDataModule, DatasetConfig, DefaultCollator
from mttl.models.library.expert_library import DatasetLibrary


@dataclass
class CLIPExpertsConfig(DatasetConfig):
    pass


@dataclass
class DataCollatorForCLIPExperts(DefaultCollator):
    def __call__(self, batch):
        sources = [b["input_texts"] for b in batch]
        accuracy = [b["score"] for b in batch]
        experts_names = [b["expert_name"] for b in batch]

        return {
            "sources_texts": sources,
            "accuracy": accuracy,
            "expert_names": experts_names,
        }


@dataclass
class DataCollatorForCLIPExpertsTriple(DefaultCollator):
    def __call__(self, batch):
        sources = [b["sources_texts"] for b in batch]
        positive_experts = [b["positive_expert_names"] for b in batch]
        negative_experts = [b["negative_expert_names"] for b in batch]
        eval_task = [b["eval_task"] for b in batch]

        return {
            "sources_texts": sources,
            "positive_expert_names": positive_experts,
            "negative_expert_names": negative_experts,
            "eval_task": eval_task,
        }


def maybe_filter_hf_dataset_by_key(dataset, key, task_names: str = None, n_proc=16):
    """Filter a HuggingFace dataset by task names."""
    # get the tasks
    key_names = set(dataset["train"][key])

    if task_names:
        task_names = sorted(task_names.split(","))
        if not set(task_names).issubset(key_names):
            raise ValueError(
                "task_names must be a subset of the available tasks. Got {} and {}".format(
                    task_names, key_names
                )
            )

    train_dataset, dev_dataset, test_dataset = None, None, None

    if task_names is not None:
        train_dataset = dataset["train"].filter(
            lambda x: x[key] in task_names,
            num_proc=n_proc,
            desc="Filtering task names",
        )
        if "validation" in dataset:
            dev_dataset = dataset["validation"].filter(
                lambda x: x[key] in task_names,
                num_proc=n_proc,
                desc="Filtering task names",
            )
        if "test" in dataset:
            test_dataset = dataset["test"].filter(
                lambda x: x[key] in task_names,
                num_proc=n_proc,
                desc="Filtering task names",
            )
    else:
        train_dataset = dataset["train"]
        if "validation" in dataset:
            dev_dataset = dataset["validation"]
        if "test" in dataset:
            test_dataset = dataset["test"]

    if task_names is None:
        task_names = list(key_names)

    task_to_id = {task: i for i, task in enumerate(task_names)}
    return task_names, task_to_id, train_dataset, dev_dataset, test_dataset


@dataclass
class CLIPExpertsDatamodule(DefaultDataModule):
    # The dataset format is [x, E, accuracy]
    DATA_ENV = "CLIP_DATA_DIR"

    def __init__(self, config):
        if os.environ.get(self.DATA_ENV) is None:
            raise ValueError(f"Environment variable {self.DATA_ENV} is not set. ")
        super().__init__(config)

    def setup_dataset(self):
        dataset_name = "x_e_acc_dataset"
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset(
            "json", data_files=os.environ[self.DATA_ENV], name=dataset_name
        )

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_key(
            dataset,
            key="positive_expert_names",
            task_names=self.config.finetune_task_name,
            n_proc=n_proc,
        )

        # get all the tasks_names from the dataset
        self._task_names = dataset["positive_expert_names"].unique()

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
                train_dataset, 0.1
            )
            self.test_dataset = self.dev_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return DataCollatorForCLIPExperts(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


@dataclass
class CLIPTripleDataModule(DefaultDataModule):
    # the dataset format is [task_eval, input x, positive_experts, negative_experts]
    DATA_ENV = "CLIP_TRIPLE_DATA_DIR"

    def __init__(self, config):
        if os.environ.get(self.DATA_ENV) is None:
            raise ValueError(f"Environment variable {self.DATA_ENV} is not set. ")
        super().__init__(config)

    def setup_dataset(self):
        dataset_name = "x_m1_m2_dataset"
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset(
            "json", data_files=os.environ[self.DATA_ENV], name=dataset_name
        )

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_key(
            dataset,
            key="eval_task",
            task_names=self.config.finetune_task_name,
            n_proc=n_proc,
        )

        # get all the tasks_names from the dataset
        self._task_names = list(
            set(
                train_dataset["positive_expert_names"]
                + train_dataset["negative_expert_names"]
            )
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
                train_dataset, 0.1
            )
            self.test_dataset = self.dev_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return DataCollatorForCLIPExpertsTriple(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )
