import torch
from datasets import load_dataset

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
from transformers import AutoTokenizer


IGNORE_INDEX = -100


def generate_prompt(example):
    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


def tokenize(tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def prepare_sample(example: dict, tokenizer, max_length: int, mask_inputs: bool = True):
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(
        tokenizer,
        full_prompt,
        max_length=max_length,
    )
    encoded_full_prompt_and_response = tokenize(
        tokenizer, full_prompt_and_response, max_length=max_length
    )

    # Add EOS token id explicitly.
    if encoded_full_prompt_and_response[-1].item() != tokenizer.eos_token_id:
        encoded_full_prompt_and_response = torch.cat(
            (encoded_full_prompt_and_response, torch.LongTensor([tokenizer.eos_token_id])), 0
        )

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = IGNORE_INDEX

    return {
        **example,
        "input_hash": hash_example(full_prompt),
        "instruction_hash": hash_example(example["instruction"]),
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "target_ids": labels,
    }


def tokenize(tokenizer, string: str, max_length: int) -> torch.Tensor:
    return tokenizer.encode_plus(
        string,
        truncation=True,
        padding="do_not_pad",
        max_length=max_length,
        return_tensors="pt",
    ).input_ids.squeeze(0)


class AlpacaDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        tokenizer,
        max_length,
        data_dir,
        train_on_inputs=False,
        prepare_for_generation=False,
    ):
        super().__init__()
        self.train_on_inputs = train_on_inputs
        # load the data
        self.dataset = load_dataset("yahma/alpaca-cleaned", cache_dir=data_dir)["train"]
        # each entry is "instruction", "input", "output" dictionary
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prepare_for_generation = prepare_for_generation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        sample = prepare_sample(
            entry,
            self.tokenizer,
            self.max_length,
            mask_inputs=True,
        )
        ex_info = ExampleInfo(
            sample["input_ids_no_response"]
            if self.prepare_for_generation
            else sample["input_ids"],
            sample["target_ids"],
            -1,
            sample["input_hash"],
            example_id=key,
            input_text=entry["instruction"],
            instruction_hash=sample["instruction_hash"],
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions
