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
        tokenizer,
        max_input_length,
        max_output_length,
        data_dir,
        train_on_inputs=False,
        dst_path=None,
        idxs=None,
        loss_for_keywords=True,
        subset=None,
    ):
        super().__init__()
        self.dst_path = dst_path
        self.loss_for_keywords = loss_for_keywords
        self.train_on_inputs = train_on_inputs

        # load the data
        if os.getenv("AP_DATA_DIR") is not None:
            data_dir = os.getenv("AP_DATA_DIR")
        if dst_path is None:
            self.dataset = load_dataset("yahma/alpaca-cleaned", cache_dir=data_dir)[
                "train"
            ]
            if idxs is not None:
                self.dataset = self.dataset.select(idxs)
        else:
            import json
            from datasets import Dataset
            import pandas as pd

            with open(dst_path, "r") as f:
                new_dataset = json.load(f)
            for ex in new_dataset:
                if "topic_set" in ex:
                    ex["topic_set"] = str(ex["topic_set"])
                if "skill_set" in ex:
                    ex["skill_set"] = str(ex["skill_set"])
            df = pd.DataFrame(new_dataset)
            if idxs is not None:
                df = df.iloc[idxs]
            # transform into dataset
            self.dataset = Dataset.from_pandas(df)

        if subset is not None:
            self.dataset = self.dataset.select(range(subset))
        # each entry is "instruction", "input", "output" dictionary

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.dataset)

    def _tokenize_fn(self, string: str) -> Dict:
        """Tokenize a list of strings."""
        return self.tokenizer.encode_plus(
            string,
            truncation=True,
            padding="do_not_pad",
            max_length=self.max_input_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)    

    def preprocess(self, source: str, target: str) -> Dict:
        full_prompt = source
        full_prompt_and_response = source + target
        encoded_full_prompt = self._tokenize_fn(full_prompt)
        encoded_full_prompt_and_response = self._tokenize_fn(full_prompt_and_response)

        # Add EOS token id explicitly.
        if encoded_full_prompt_and_response[-1].item() != self.tokenizer.eos_token_id:
            encoded_full_prompt_and_response = torch.cat(
                (
                    encoded_full_prompt_and_response,
                    torch.LongTensor([self.tokenizer.eos_token_id]),
                ),
                0,
            )

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_full_prompt_and_response.clone()

        if not self.train_on_inputs:
            labels[: len(encoded_full_prompt)] = IGNORE_INDEX

        return {
            "input_ids": encoded_full_prompt_and_response,
            "labels": labels,
        }

    def __getitem__(self, key):
        entry = self.dataset[key]

        enc_input_for_hash = AlpacaTemplateForHash.apply(entry)
        input_hash = hash_example(enc_input_for_hash)
        instruction_hash = hash_example(entry["instruction"])
        topics_str = None

        enc_input = AlpacaTemplate.apply(entry)
        source = AlpacaTemplateSource.apply(entry)
        task_id = 0

        tok_input = self.preprocess(source, entry["output"])
        ex_info = ExampleInfo(
            tok_input["input_ids"],
            tok_input["labels"],
            task_id,
            input_hash,
            example_id=key,
            input_text=(enc_input),
            instruction_hash=instruction_hash,
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions
