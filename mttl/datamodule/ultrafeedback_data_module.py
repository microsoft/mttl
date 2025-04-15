from dataclasses import dataclass
from typing import Any
import torch

from mttl.datamodule.base import DatasetConfig, DefaultCollator, DataModule
from mttl.models.library.dataset_library import DatasetLibrary


def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(
        isinstance(message, dict) for message in messages
    ):
        return all("role" in message and "content" in message for message in messages)
    return False


@dataclass
class UltrafeedbackDPOCollator(DefaultCollator):
    def __call__(self, batch):

        # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
        # We therefore need to extract the N-1 turns to form the prompt
        prompts = []
        chosen_responses = []
        rejected_responses = []
        for example in batch:
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]
            prompts.append(
                self.tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            )
            chosen_responses.append(
                self.tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            )
            rejected_responses.append(
                self.tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            )

        prompt_ids = self.tokenizer.batch_encode_plus(
            prompts,
            padding=True,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )["input_ids"]

        prefered_tokenize = self.tokenizer.batch_encode_plus(
            chosen_responses,
            padding=True,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )
        prefered_ids = prefered_tokenize["input_ids"]

        disprefered_tokenize = self.tokenizer.batch_encode_plus(
            rejected_responses,
            padding=True,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        )
        disprefered_ids = disprefered_tokenize["input_ids"]

        prompt_prefered_ids = torch.cat([prompt_ids, prefered_ids], dim=-1)
        prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)

        prompt_prefered_mask = torch.cat(
            [torch.ones_like(prompt_ids), torch.zeros_like(prefered_ids)], dim=-1
        )
        # compute the each length of the prefered
        prefered_y_len = prefered_tokenize["attention_mask"].sum(dim=1)
        disprefered_y_len = disprefered_tokenize["attention_mask"].sum(dim=1)

        prompt_disprefered_mask = torch.cat(
            [torch.ones_like(prompt_ids), torch.zeros_like(disprefered_ids)], dim=-1
        )

        return {
            "prompt_prefered_ids": prompt_prefered_ids,
            "prompt_disprefered_ids": prompt_disprefered_ids,
            "prompt_prefered_mask": prompt_prefered_mask,
            "prompt_disprefered_mask": prompt_disprefered_mask,
            "prefered_y_len": prefered_y_len,
            "disprefered_y_len": disprefered_y_len,
        }


@dataclass
class UltrafeedbackDPOmodule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(
            "princeton-nlp/gemma2-ultrafeedback-armorm"
        )

        # format the ultrafeedback dataset to chatbot format
        self.train_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        self.dev_dataset = self.test_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return UltrafeedbackDPOCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


@dataclass
class UltrafeedbackSFTCollator(DefaultCollator):
    def __call__(self, batch):

        # For SFT, the inputs are triples of (prompt, message), where `chosen` and `rejected` are the final turn of a dialogue
        # We therefore need to extract the N-1 turns to form the prompt
        prompts = []
        messages = []
        for example in batch:
            prompt_messages = example["prompt"]
            chosen_messages = example["messages"]
            prompts.append(prompt_messages)
            messages.append(
                self.tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            )

        return {
            "sources_texts": prompts,
            "labels_texts": messages,
        }


@dataclass
class UltrafeedbackSFTmodule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry("HuggingFaceH4/ultrachat_200k")

        # format the ultrafeedback dataset to chatbot format
        self.train_dataset = dataset["train_sft"]
        self.test_dataset = dataset["test_sft"]
        self.dev_dataset = self.test_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return UltrafeedbackSFTCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


if __name__ == "__main__":
    config = DatasetConfig(model="microsoft/Phi-3-mini-4k-instruct")
    datamodule = UltrafeedbackSFTmodule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        # prompt_prefered_mask = batch["prompt_prefered_mask"]
        # prompt_disprefered_mask = batch["prompt_disprefered_mask"]

        # get the length of the response
        # prefered_y_len = batch["prefered_y_len"]
        # disprefered_y_len = batch["disprefered_y_len"]
        print(batch)
        breakpoint()
