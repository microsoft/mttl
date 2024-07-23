import os
from dataclasses import dataclass

import torch
from datasets import concatenate_datasets, get_dataset_split_names

from mttl.datamodule.base import DefaultDataModule
from mttl.datamodule.platypus_module import PlatypusConfig
from mttl.logging import logger, setup_logging
from mttl.models.library.expert_library import DatasetLibrary


@dataclass
class FactsCollator:
    def __call__(self, batch, **kwargs):
        batch = [b[0] for b in batch]
        output_batch = {}
        output_batch["input_ids"] = torch.stack(batch, 0)
        output_batch["attention_mask"] = torch.ones_like(output_batch["input_ids"])
        output_batch["labels"] = torch.stack(batch, 0)
        return output_batch


def _load_dataset(dataset, split):
    dataset = DatasetLibrary.pull_dataset(dataset)
    if split in dataset:
        return dataset[split]

    dataset = dataset["train"]
    # filter "subject" column
    dataset = dataset.filter(lambda x: x["subject"] == split)
    return dataset


@dataclass
class FactsLMConfig(PlatypusConfig):
    text_field: str = "facts"


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
            datasets_.append(_load_dataset(self.config.dataset, split=task_name))
        self.dataset = concatenate_datasets(datasets_)

        train_facts = []
        valid_facts = []

        for example in self.dataset:
            facts = example[self.config.text_field]
            facts = self.tokenizer(facts)["input_ids"]
            train_facts.extend(
                facts[: -int(self.config.validation_portion * len(facts))]
                + [self.tokenizer.eos_token_id]
            )
            valid_facts.extend(
                facts[-int(self.config.validation_portion * len(facts)) :]
                + [self.tokenizer.eos_token_id]
            )

        def form_dataset(data):
            data = [
                torch.tensor(data[i : i + self.config.max_input_length])
                for i in range(0, len(data), self.config.max_input_length)
            ]
            if len(data) > 1:
                data[-1] = torch.concatenate(
                    (
                        data[-2][-self.config.max_input_length + len(data[-1]) :],
                        data[-1],
                    )
                )
            return torch.stack(data, 0)

        self.train_dataset = torch.utils.data.TensorDataset(form_dataset(train_facts))
        self.dev_dataset = self.test_dataset = torch.utils.data.TensorDataset(
            form_dataset(valid_facts)
        )

        logger.info("Training examples: {}".format(len(self.train_dataset)))
        logger.info("Validation examples: {}".format(len(self.dev_dataset)))


if __name__ == "__main__":
    from mttl.config import Config

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
