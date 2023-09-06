import torch
import os
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Iterator, List, Sized, Union, Optional
from dataclasses import dataclass

from mttl.datamodule.mmlu_data_module import MMLUDataModule
from mttl.datamodule.utils import get_tokenizer
from mttl.utils import logger


@dataclass
class WikiMMLUDataCollator:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_input_length: Optional[int] = 2048
    max_output_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    model_family: str = "gpt"
    task_to_id: dict = None
    rng: np.random.RandomState = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        if self.model_family != "gpt":
            raise ValueError("Only GPT is supported as data family for Wiki-MMLU.")

        num_words = int(self.max_input_length * 0.75)

        # chunk approx 75% of the text
        sources = []
        for instance in batch:
            split_text = instance["text"].split(" ")
            if len(split_text) <= num_words:
                sources.append(" ".join(split_text))
            else:
                split_point = self.rng.randint(0, len(split_text) - num_words)
                sources.append(
                    " ".join(split_text[split_point : split_point + num_words])
                )

        task_names = [instance["subject"] for instance in batch]
        output_batch = self.tokenizer(
            sources,
            max_length=self.max_input_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        output_batch["task_names"] = task_names
        if self.task_to_id is not None:
            output_batch["task_ids"] = torch.LongTensor(
                [self.task_to_id[task] for task in task_names]
            )
        labels = output_batch["input_ids"]
        output_batch["labels"] = torch.masked_fill(
            labels, ~output_batch["attention_mask"].bool(), self.label_pad_token_id
        )
        return output_batch


class TaskSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, task_names, batch_size, num_tasks_per_batch=1, rng=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_tasks_per_batch = num_tasks_per_batch
        self.num_examples = len(self.dataset)
        self.num_tasks = len(set(task_names))
        self.rng = rng or np.random.RandomState(0)

        tasks_to_ids = {task: i for i, task in enumerate(set(task_names))}
        self.task_ids = [tasks_to_ids[task] for task in task_names]

    def __len__(self):
        import math
        return self.batch_size * math.ceil(self.num_examples / self.batch_size)

    def __iter__(self):
        active_examples = [set() for _ in range(self.num_tasks)]
        for i in range(self.num_examples):
            active_examples[self.task_ids[i]].add(i)
        active_tasks = [i for i in range(self.num_tasks) if len(active_examples[i])]

        for j in range(0, self.num_examples):
            if j % (self.batch_size // self.num_tasks_per_batch) == 0:
                t = self.rng.choice(active_tasks)

            ex_pool = active_examples[t]
            a = self.rng.choice(list(ex_pool))

            yield a
            ex_pool.remove(a)

            if not len(ex_pool):
                active_tasks.remove(t)

            if not len(active_tasks):
                break


class WikiMMLUDataModule(LightningDataModule):
    def get_task_sampler(self, dataset, batch_size):
        return TaskSampler(
            dataset=dataset,
            task_names=self.train_tasks,
            batch_size=batch_size,
            num_tasks_per_batch=batch_size,
            rng=self.rng,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
            sampler=self.get_task_sampler(
                self.train_dataset,
                batch_size=self.config.train_batch_size,
            )
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tokenizer = get_tokenizer(config)

        self.rng = np.random.RandomState(config.seed)
        self.mmlu_module = MMLUDataModule(config, data_dir="from_env")
        self.setup_dataset()

    def setup_dataset(self, stage=None):
        dataset = load_dataset(self.config.dataset)

        self.task_to_id = self.mmlu_module.task_to_id
        self.collate_fn = WikiMMLUDataCollator(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family="gpt",
            task_to_id=self.task_to_id,
            rng=self.rng,
        )

        torch_rng = torch.Generator().manual_seed(self.config.seed)

        self.train_dataset = dataset["train"]
        n_tr_samples = int(len(self.train_dataset) * (1 - 0.03))
        self.train_dataset, self.dev_dataset = torch.utils.data.random_split(
            self.train_dataset,
            [
                n_tr_samples,
                len(self.train_dataset) - n_tr_samples,
            ],
            generator=torch_rng,
        )
        self.test_dataset = self.dev_dataset
        self.train_tasks = [i["subject"] for i in self.train_dataset]

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
    config.dataset = "sordonia/wiki_mmlu_1M"
    config.model_family = "gpt"
    os.environ["MMLU_DATA_DIR"] = "/datadrive/datasets/mmlu/data"

    datamodule = WikiMMLUDataModule(config)
    batch = next(iter(datamodule.train_dataloader()))
    breakpoint()
