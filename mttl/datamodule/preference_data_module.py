from dataclasses import dataclass

import torch

from mttl.datamodule.base import DatasetConfig, DefaultCollator, DataModule
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class DataCollatorForDPO(DefaultCollator):
    def __call__(self, batch):
        prompts = ["Instruct: " + item["prompt"] + "\n" for item in batch]
        chosen_responses = ["Output: " + item["chosen"] for item in batch]
        rejected_responses = ["Output: " + item["rejected"] for item in batch]

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
class Preferencemodule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):
        train_dataset = DatasetLibrary.pull_dataset_with_retry(
            "jondurbin/truthy-dpo-v0.1"
        )["train"]

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset, 0.1
        )
        self.test_dataset = self.dev_dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return DataCollatorForDPO(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


if __name__ == "__main__":
    config = DatasetConfig(model="microsoft/phi-2")
    datamodule = Preferencemodule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        prompt_prefered_mask = batch["prompt_prefered_mask"]
        prompt_disprefered_mask = batch["prompt_disprefered_mask"]

        # get the length of the response
        prefered_y_len = batch["prefered_y_len"]
        disprefered_y_len = batch["disprefered_y_len"]
        print(prefered_y_len, disprefered_y_len)
        breakpoint()
