import torch
from datasets import (
    concatenate_datasets,
    get_dataset_config_names,
    Dataset,
)
import numpy as np

from mttl.models.library.expert_library import DatasetLibrary


class PlatypusTemplate:
    @classmethod
    def apply(self, instruction, input=None):
        if input is not None and len(input) > 0:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt


class InversePlatypusTemplate:
    @classmethod
    def apply(self, output, input=None, icl_examples=None):
        if input is not None and len(input):
            prompt = f"Below is a response to a task, paired with an input that provides further context. Write an instruction that appropriately describes the response.\n\n### Input:\n{input}\n\n### Response:\n{output}\n\n### Instruction:\n"
        else:
            prompt = f"Below is a response to a task. Write an instruction that appropriately describes the response.\n\n### Response:\n{output}\n\n### Instruction:\n"

        if icl_examples is not None:
            icl_prompt = f"Here are some examples of good instructions that you should imitate:\n"
            for icl_example in icl_examples:
                icl_prompt += f"\n### Instruction:\n{icl_example}"
            icl_prompt += "\n\n"
            return icl_prompt + prompt
        else:
            return prompt


def preprocess(mix_in: Dataset):
    import pandas as pd

    if mix_in is None:
        return None

    mapping = {
        "Task": "subject",
        "Input": "instruction",
        "Output": "response",
    }
    new_dsts = {}
    unique_subjects = np.unique(mix_in["Task"])
    for subject in unique_subjects:
        samples = []
        for sample in mix_in.filter(lambda x: x["Task"] == subject):
            new_sample = {mapping[k]: v for k, v in sample["Instance"].items()}
            new_sample["subject"] = subject
            samples.append(new_sample)
            positive_examples = sample["Positive Examples"]
            example_samples = positive_examples.split("\n\n")
            for ex in example_samples:
                if len(ex) == 0:
                    continue
                ex = {
                    "instruction": ex.split("\nAnswer:")[0],
                    "response": ex.split("\nAnswer:")[1],
                }
                if ex not in samples:
                    samples.append(ex)
        new_dsts[subject] = Dataset.from_pandas(pd.DataFrame(data=samples))
    return new_dsts


class PlatypusDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_name: str = "garage-bAInd/Open-Platypus"):
        super().__init__()

        self.dataset = DatasetLibrary.pull_dataset(dataset_name, split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        source = PlatypusTemplate.apply(entry["instruction"], entry["input"])
        target = entry["output"]

        return {
            "source": source,
            "target": target,
            "example_id": key,
            "instruction": entry.get("instruction"),
            "data_source": entry["data_source"],
        }

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions


class PlatypusQADataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        dataset_name: str = None,
        filter_by_subject: str = None,
        val_mixin: Dataset = None,
    ):
        super().__init__()

        if filter_by_subject is not None:
            task_names = sorted(filter_by_subject.split(","))
        else:
            task_names = get_dataset_config_names(dataset_name)

        datasets_ = []
        for task_name in task_names:
            datasets_.append(DatasetLibrary.pull_dataset(dataset_name, split=task_name))

        self.dataset = concatenate_datasets(datasets_)
        self.val_mixin = preprocess(val_mixin)
        self.mixin_idxs = None

        if self.val_mixin is not None:
            if filter_by_subject is not None:
                val_mixins = [self.val_mixin[sub] for sub in task_names]
            else:
                val_mixins = [v for k, v in self.val_mixin.items()]

            self.val_mixin = concatenate_datasets(val_mixins)
            self.dataset = concatenate_datasets([self.dataset, self.val_mixin])
            len_mixin = len(self.val_mixin)
            self.mixin_idxs = torch.arange(len(self.dataset))[-len_mixin:]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        source = PlatypusTemplate.apply(entry["instruction"], entry.get("input"))
        target = entry["response"] if "response" in entry else entry["output"]

        return {
            "source": source,
            "target": target,
            "example_id": key,
            "instruction": entry.get("instruction"),
        }

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions


class InversePlatypusDataset(PlatypusDataset):
    def __getitem__(self, key):
        entry = self.dataset[key]

        source = InversePlatypusTemplate.apply(
            entry["output"], entry.get("input"), entry.get("icl_examples")
        )
        target = entry["instruction"]

        return {
            "source": source,
            "target": target,
            "example_id": key,
            "instruction": entry.get("instruction"),
        }
