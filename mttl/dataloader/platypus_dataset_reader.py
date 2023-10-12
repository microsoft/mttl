import torch
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example, logger


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


class PlatypusDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self, data_dir: str = None, dataset_name: str = "garage-bAInd/Open-Platypus"
    ):
        super().__init__()
        self.dataset = load_dataset(dataset_name)["train"]
        logger.info(self[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        source = PlatypusTemplate.apply(
            entry["instruction"], entry["input"]
        )
        labels = entry["output"]
        hash = hash_example(source)
        instruction_hash = hash_example(entry["instruction"])

        ex_info = ExampleInfo(
            source,
            labels,
            task_id=-1,
            example_id=key,
            input_text=source,
            hash=hash,
            instruction_hash=instruction_hash,
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions


class PlatypusQADataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        data_dir: str = None,
        dataset_name: str = None,
        filter_by_subject: str = None,
    ):
        super().__init__()

        if filter_by_subject is not None:
            task_names = sorted(filter_by_subject.split(","))
        else:
            task_names = get_dataset_config_names(dataset_name)

        datasets_ = []
        for task_name in task_names:
            datasets_.append(load_dataset(dataset_name, split=task_name))
        self.dataset = concatenate_datasets(datasets_)
        logger.info(self[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        source = PlatypusTemplate.apply(entry["instruction"], entry.get("input"))
        labels = entry["response"] if "response" in entry else entry["output"]
        hash = hash_example(source)
        instruction_hash = hash_example(entry["instruction"])

        ex_info = ExampleInfo(
            source,
            labels,
            task_id=-1,
            example_id=key,
            input_text=source,
            hash=hash,
            instruction_hash=instruction_hash,
        )
        return ex_info

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
        labels = entry["instruction"]
        hash = hash_example(source)
        instruction_hash = hash_example(entry["instruction"])

        ex_info = ExampleInfo(
            source,
            labels,
            task_id=-1,
            example_id=key,
            input_text=source,
            hash=hash,
            instruction_hash=instruction_hash,
        )
        return ex_info
