import os
from dataclasses import dataclass
from functools import partial

import numpy

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class MBPPDataConfig(DatasetConfig):
    name: str = "sanitized"
    use_instruct_template: bool = False


def instruct_template(example):
    template = "# Instruct:\n# These are the assertions for your function: {}\n".format(
        "\n".join(example["test_list"])
    )
    template += "'''{}'''\n".format(example["prompt"])
    target = example["code"]

    example["task_source"] = "mbpp"
    example["task_name"] = "mbpp"
    example["source"] = template
    example["target"] = target
    example["code_prefix"] = example["source"]
    example["code_tests"] = "\n".join(example["test_list"])
    return example


def detect_indentation(func):
    source_lines = func.splitlines()
    # Skip function definition line to check indentation of the body
    for line in source_lines:
        if line.strip() and not line.strip().startswith("#"):  # Skip empty lines
            # Count spaces at the beginning of the line
            space_count = len(line) - len(line.lstrip())
            if line.startswith("\t"):
                return "\t"
            if space_count > 0:
                return " " * space_count
    return None


def completion_template(for_generation, example):
    """Format the MBPP dataset into source and target."""
    example["task_source"] = "mbpp"
    example["task_name"] = "mbpp"

    # format the code and test cases
    code_header = example["code"].partition(":")[0] + ":"
    code_body = example["code"].partition(":")[2].lstrip("\n")

    if for_generation:
        # use tab for indentation when generating code
        indent = "\t"
    else:
        # we need to match the indentation used in the code
        # to ensure that source and target are aligned nicely and
        # executable
        indent = detect_indentation(code_body)

    # the format of the source is:
    # def function_name(arg1, arg2):  (code_header)
    # (indent) """
    #          prompt
    #          list of assertions
    #          """
    source = example["prompt"] if "prompt" in example else example["text"]
    source_template = '{}\n{}"""\n{}{}\n{}{}\n{}"""\n'
    example["source"] = source_template.format(
        code_header,
        indent,
        indent,
        source,
        indent,
        f"\n{indent}".join(example["test_list"]),
        indent,
    )
    # we cannot use the code as target because it is not formatted correctly for completion
    example["target"] = code_body
    example["code_prefix"] = example["source"]
    example["code_tests"] = "\n".join(example["test_list"])
    return example


@DataModule.register("mbpp", config_cls=MBPPDataConfig)
class MBPPDataModule(DataModule):
    collate_extra_fields = ["code_prefix", "code_tests"]

    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset("mbpp", name=self.config.name)

        dataset = dataset.map(
            (
                instruct_template
                if self.config.use_instruct_template
                else partial(completion_template, self.for_generation)
            ),
            num_proc=n_proc,
        )

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            valid_dataset,
            test_dataset,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        self.train_dataset = train_dataset
        self.dev_dataset = valid_dataset
        self.test_dataset = test_dataset


def code_exercises_template(example):
    """Format the MBPP dataset into source and target."""
    example["task_source"] = "code-ex"
    example["task_name"] = "code-ex"
    # we cannot use the code as target because it is not formatted correctly for completion
    example["source"] = example["problem"]
    example["code_prefix"] = example["source"]
    example["target"] = example["solution"]
    return example


@dataclass
class CodeExDataConfig(DatasetConfig):
    pass


class CodeExDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        dataset = DatasetLibrary.pull_dataset("jinaai/code_exercises")
        dataset = dataset["train"].select(range(10000))

        def create_split(rng, _):
            return {"split": rng.choice(["train", "valid"], p=[0.9, 0.1])}

        dataset = dataset.map(
            partial(create_split, numpy.random.RandomState(42)),
            num_proc=n_proc,
            desc="Creating split column.",
        )

        dataset = dataset.map(
            code_exercises_template,
            num_proc=n_proc,
        )

        self.train_dataset = dataset.filter(
            lambda x: x["split"] == "train",
            num_proc=n_proc,
            desc="Creating train set",
        )
        self.dev_dataset = dataset.filter(
            lambda x: x["split"] in ["validation", "valid"],
            num_proc=n_proc,
            desc="Creating valid set",
        )
        self.test_dataset = dataset.filter(
            lambda x: x["split"] == "test",
            num_proc=n_proc,
            desc="Creating test set",
        )

        if len(self.test_dataset) == 0:
            self.test_dataset = self.dev_dataset
