import torch

from mttl.logging import logger
from mttl.models.library.dataset_library import DatasetLibrary
from mttl.utils import hash_example


class Oasst1Template:
    @classmethod
    def apply(self, dict_values):
        instruction, output = (
            dict_values["instruction"],
            dict_values["response"],
        )
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"


class InverseOasst1Template:
    @classmethod
    def apply(self, dict_values):
        instruction, output = (
            dict_values["instruction"],
            dict_values["response"],
        )
        return f"Below is a response to a task. Write an instruction that appropriately describes the response.\n\n### Response:\n{output}\n\n### Instruction:\n"


class Oasst1Dataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        data_dir: str = None,
    ):
        super().__init__()
        self.dataset = DatasetLibrary.pull_dataset(
            "ostapeno/oasst1_seed3200", split="train"
        )
        logger.info(self[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        source = Oasst1Template.apply(entry)
        labels = entry["response"]

        return {
            "source": source,
            "target": labels,
            "example_id": key,
            "instruction": entry.get("instruction"),
        }

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions


class InverseOasst1Dataset(Oasst1Dataset):
    def __getitem__(self, key):
        entry = self.dataset[key]

        source = InverseOasst1Template.apply(entry)
        labels = entry["instruction"]
        hash = hash_example(source)
        instruction_hash = hash_example(entry["instruction"])

        return {
            "source": source,
            "target": labels,
            "example_id": key,
            "instruction": entry.get("response"),
        }
