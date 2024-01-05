from datasets import load_dataset
from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from dataclasses import dataclass
import os

from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.datamodule.mt_seq_to_seq_module import apply_source_template


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


def completion_template(example):
    """Format the MBPP dataset into source and target."""
    example["task_source"] = "mbpp"
    example["task_name"] = "mbpp"

    # format the code and test cases
    code_header = example["code"].partition(":")[0] + ":"
    code_body = example["code"].partition(":")[2].lstrip("\n")
    code_body = code_body.replace("    ", "\t")
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


class MBPPDataModule(DefaultDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = load_dataset("mbpp", name=self.config.name)

        dataset = dataset.map(
            instruct_template
            if self.config.use_instruct_template
            else completion_template,
            num_proc=n_proc,
            remove_columns=["task_id"],
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
