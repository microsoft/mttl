import os
from dataclasses import dataclass

from mttl.datamodule.base import DefaultDataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.models.library.expert_library import DatasetLibrary


@dataclass
class HumanEvalConfig(DatasetConfig):
    use_instruct_template: bool = False


# convert task_id to task_name and labels
def instruct_template(example):
    # there are not test lists in human eval, so we have to create them
    import re

    prompt = example["prompt"]
    try:
        regex = r'"""([\S\n\t\v ]*)"""'
        query = re.findall(regex, prompt)[0].strip()
    except:
        regex = r"'''([\S\n\t\v ]*)'''"
        query = re.findall(regex, prompt)[0]
    function_name = example["entry_point"].strip()

    assertion = (
        example["test"].split("assert")[1].strip().replace("candidate", function_name)
    )

    template = "Instruct:\n# These are the assertions for your function: {}\n".format(
        assertion
    )
    template += "'''{}'''\n".format(query)
    template += "Answer:\n"

    target = example["prompt"] + example["canonical_solution"]
    example["task_source"] = "humaneval"
    example["task_name"] = "humaneval"
    example["source"] = template
    example["target"] = target
    example["code_prefix"] = example["source"]
    example["code_tests"] = example["test"] + "\n" + f"check({example['entry_point']})"
    return example


def completion_template(example):
    example["task_source"] = "humaneval"
    example["task_name"] = "humaneval"
    example["source"] = example["prompt"].lstrip()
    example["target"] = example["canonical_solution"]
    example["code_prefix"] = example["prompt"].lstrip()
    example["code_tests"] = example["test"] + "\n" + f"check({example['entry_point']})"
    return example


class HumanEvalDataModule(DefaultDataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        dataset = DatasetLibrary.pull_dataset(
            "openai_humaneval", name="openai_humaneval"
        )

        dataset = dataset.map(
            instruct_template
            if self.config.use_instruct_template
            else completion_template,
            num_proc=n_proc,
            remove_columns=["prompt", "test", "entry_point", "task_id"],
        )

        (
            self._task_names,
            self._task_to_id,
            _,
            _,
            test_dataset,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        self.train_dataset = self.dev_dataset = self.test_dataset = test_dataset
