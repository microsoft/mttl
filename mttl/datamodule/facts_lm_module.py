import torch
import os
import numpy as np
from datasets import load_dataset, get_dataset_split_names, concatenate_datasets
from typing import Optional
from dataclasses import dataclass
from mttl.datamodule.platypus_module import PlatypusConfig
from mttl.datamodule.collators import DefaultDataModule

from mttl.utils import logger


@dataclass
class FactsCollator:
    def __call__(self, batch, **kwargs):
        batch = [b[0] for b in batch]
        output_batch = {}
        output_batch["input_ids"] = torch.stack(batch, 0)
        output_batch["attention_mask"] = torch.ones_like(output_batch["input_ids"])
        output_batch["labels"] = torch.stack(batch, 0)
        return output_batch


@dataclass
class FactsLMConfig(PlatypusConfig):
    pass


class FactsLMDataModule(DefaultDataModule):
    @property
    def collate_fn(self):
        return FactsCollator()

    def setup_dataset(self, stage=None):
        if self.config.finetune_task_name is not None:
            task_names = sorted(self.config.finetune_task_name.split(","))
        else:
            task_names = get_dataset_split_names(self.config.dataset)

        datasets_ = []
        for task_name in task_names:
            datasets_.append(load_dataset(self.config.dataset, split=task_name))
        self.dataset = concatenate_datasets(datasets_)

        train_facts = []
        valid_facts = []
        for example in self.dataset:
            facts = example["facts"]
            facts = facts.split("\n")
            train_facts.extend(
                facts[: -int(self.config.validation_portion * len(facts))]
            )
            valid_facts.extend(
                facts[-int(self.config.validation_portion * len(facts)) :]
            )

        train_facts = "\n".join(train_facts)
        valid_facts = "\n".join(valid_facts)

        # split into chunks
        train_tokenized = self.tokenizer(train_facts)["input_ids"]
        valid_tokenized = self.tokenizer(valid_facts)["input_ids"]
        train_data = [
            torch.tensor(train_tokenized[i : i + self.config.max_input_length])
            for i in range(0, len(train_tokenized), self.config.max_input_length)
        ]

        train_data[-1] = torch.concatenate(
            (
                train_data[-2][-self.config.max_input_length + len(train_data[-1]) :],
                train_data[-1],
            )
        )
        valid_data = [
            torch.tensor(valid_tokenized[i : i + self.config.max_input_length])
            for i in range(0, len(valid_tokenized), self.config.max_input_length)
        ]
        valid_data[-1] = torch.concatenate(
            (
                valid_data[-2][-self.config.max_input_length + len(valid_data[-1]) :],
                valid_data[-1],
            )
        )

        self.train_dataset = torch.utils.data.TensorDataset(torch.stack(train_data, 0))
        self.dev_dataset = self.test_dataset = torch.utils.data.TensorDataset(
            torch.stack(valid_data, 0)
        )

        logger.info("Training examples: {}".format(len(self.train_dataset)))
        logger.info("Validation examples: {}".format(len(self.dev_dataset)))


if __name__ == "__main__":
    import os
    from mttl.config import Config
    from mttl.utils import setup_logging

    setup_logging()

    config = Config()
    config.max_input_length = 4096
    config.model = "meta-llama/Llama-2-7b-hf"
    config.dataset = "sordonia/facts-text-davinci-003_clen128_maxD100_maxC-1"
    config.model_family = "gpt"
    os.environ["MMLU_DATA_DIR"] = "/datadrive/datasets/mmlu/data"

    datamodule = FactsLMDataModule(config)
    batch = next(iter(datamodule.train_dataloader()))
    breakpoint()
