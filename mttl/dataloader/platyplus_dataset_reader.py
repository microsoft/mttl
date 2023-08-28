import copy
import json
import torch
import os
import numpy as np
import transformers
from datasets import load_dataset
from scipy.stats import entropy as calc_entropy

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
from typing import List, Sequence, Dict


IGNORE_INDEX = -100


class AlpacaTemplateForHash(
    object
):  # dont change it to keep compatibility with old clusterings etc., previously generated hashes
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


class AlpacaTemplate(AlpacaTemplateForHash):
    # same as AlpacaTemplateForHash
    pass


class AlpacaTemplateSource(AlpacaTemplate):
    pass


class PlatypusDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        data_dir,
        idxs=None,
        loss_for_keywords=True,
        subset=None,
    ):
        super().__init__()
        self.loss_for_keywords = loss_for_keywords

        # load the data
        if os.getenv("AP_DATA_DIR") is not None:
            data_dir = os.getenv("AP_DATA_DIR")

        self.dataset = load_dataset("garage-bAInd/Open-Platypus", cache_dir=data_dir)[
            "train"
        ]
        if idxs is not None:
            self.dataset = self.dataset.select(idxs)
        if subset is not None:
            self.dataset = self.dataset.select(range(subset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        enc_input_for_hash = AlpacaTemplateForHash.apply(entry)
        input_hash = hash_example(enc_input_for_hash)
        instruction_hash = hash_example(entry["instruction"])

        source = AlpacaTemplateSource.apply(entry)
        enc_input = f"{source}{entry['output']}"

        task_id = -1
        ex_info = ExampleInfo(
            source,
            entry["output"],
            task_id,
            input_hash,
            example_id=key,
            input_text=enc_input,
            instruction_hash=instruction_hash,
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions
