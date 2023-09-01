import torch
import os
from datasets import load_dataset

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example


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
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Input:{input}\
            \n### Response: {output}"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Response: {output}"


class AlpacaTemplate(object):
    @classmethod
    def apply(self, dict_values, topics_str=None):
        instruction, input, output = (
            dict_values["instruction"],
            dict_values["input"],
            dict_values["output"],
        )
        if len(input) > 0:
            instr = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Input: {input}"
        else:
            instr = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}"
        # if topics_str is not None:
        #         instr += f"\n###Response: [ {topics_str} ] {output}"
        # else:
        instr += f"\n### Response: {output}"
        return instr


class AlpacaTemplateSource(object):
    @classmethod
    def apply(self, dict_values):
        instruction, input, output = (
            dict_values["instruction"],
            dict_values["input"],
            dict_values["output"],
        )
        if len(input) > 0:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Input: {input}\
            \n### Response:"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Response:"


class AlpacaDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,   
        data_dir,
        idxs=None,
        subset=None,
    ):
        super().__init__()     
        self.dataset = load_dataset("yahma/alpaca-cleaned")["train"] #, cache_dir=data_dir)["train"]

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

        enc_input = AlpacaTemplate.apply(entry)
        input = AlpacaTemplateSource.apply(entry)
        target = entry["output"]
        task_id = 0

        ex_info = ExampleInfo(
            input,
            target,
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
