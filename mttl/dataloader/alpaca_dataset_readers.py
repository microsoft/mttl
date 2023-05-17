import torch    
from datasets import load_dataset

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example


class AlpacaTemplate(object):
    @classmethod
    def apply(self, dict_values):
        instruction, input, output = dict_values["instruction"], dict_values["input"], dict_values["output"]
        if len(input)>0:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Input:{input}\
            \n### Response: {output}"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Response: {output}"


class AlpacaDataset(torch.utils.data.dataset.Dataset):     
    def __init__(self, tokenizer, max_input_length, max_output_length):
        super().__init__()

        # load the data 
        self.dataset = load_dataset("yahma/alpaca-cleaned")["train"]
        # each entry is "instruction", "input", "output" dictionary

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]
        # really basic template for now
        # TODO: check with AS if this is OOP approved
        enc_input = AlpacaTemplate.apply(entry)
        # dec_input = entry["output"]

        # next we tokenize      
        tok_input = self.tokenizer(
            enc_input, 
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        input_hash = hash_example(enc_input)
        instruction_hash = hash_example(entry["instruction"])

        ex_info = ExampleInfo(
            tok_input,
            tok_input,
            -1,
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
