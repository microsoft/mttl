import torch
from datasets import load_dataset

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example, logger


class PlatypusTemplate:
    @classmethod
    def apply(self, dict_values):
        instruction, input, output = (
            dict_values["instruction"],
            dict_values["input"],
            dict_values["output"],
        )
        if len(input) > 0:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"


class InversePlatypusTemplate:
    @classmethod
    def apply(self, dict_values):
        instruction, input, output = (
            dict_values["instruction"],
            dict_values["input"],
            dict_values["output"],
        )
        prompt = ""
        if (
            "icl_examples" in dict_values.keys()
            and dict_values["icl_examples"] is not None
        ):
            icl_examples = dict_values["icl_examples"]
            prompt += f"Here are some examples of good instructions that you should imitate:"
            for icl_example in icl_examples:
                prompt += f"\n### Instruction:\n{icl_example}"
            prompt += "\n\n"
        if instruction is None:
            prompt += f"\nBelow is a response to a task. Write an instruction that appropriately describes the response.\n\n### Response:\n{output}\n\n### Instruction:\n"
            return prompt
        else:
            # treat instruction as old instructions that we want to impove
            prompt += f"\nBelow is an old instruction and a response to a task. Write a better instruction that appropriately describes the response.\n\n### Old instruction:\n{instruction}\n\n### Response:\n{output}\n\n### Instruction:\n"
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

        source = PlatypusTemplate.apply(entry)
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

        self.dataset = load_dataset(dataset_name)["train"]

        task_names = set(list(self.dataset["subject"]))
        if filter_by_subject is not None:
            task_subset = sorted(filter_by_subject.split(","))
            if any(task not in task_names for task in task_subset):
                raise ValueError("Unknown subject name.")

            task_names = task_subset
            self.dataset = self.dataset.filter(lambda x: x["subject"] in task_names)
        logger.info(self[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        source = PlatypusTemplate.apply(entry)
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


class InversePlatypusDataset(PlatypusDataset):
    def __getitem__(self, key):
        entry = self.dataset[key]

        source = InversePlatypusTemplate.apply(entry)
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
