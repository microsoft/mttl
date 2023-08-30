import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Union, Optional

from datasets import load_dataset
from dataclasses import dataclass

from mttl.datamodule.utils import get_tokenizer
from mttl.datamodule.collators import DefaultCollator


@dataclass
class DataCollatorForMMLU(DefaultCollator):
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_input_length: Optional[int] = 2048
    max_output_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    model_family: str = "seq2seq"
    task_to_id: dict = None
    counter: int = 0

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []

        for instance in batch:
            prompt = (
                instance["Definition"]
                + instance["Positive Examples"]
                + instance["Instance"]["Input"]
            )
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

            while input_ids.shape[-1] > self.max_input_length and len(instance["Positive Examples"].split("\n\n")) > 2:
                instance["Positive Examples"] = (
                    "\n\n".join(instance["Positive Examples"].split("\n\n")[:-2])
                    + "\n\n"
                )
                prompt = (
                    instance["Definition"]
                    + instance["Positive Examples"]
                    + instance["Instance"]["Input"]
                )
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

            sources.append(prompt)

        # Remove multiple spaces, which mess with tiktoken
        labels = [instance["Instance"]["Output"] for instance in batch]
        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        task_names = [instance["Task"] for instance in batch]
        output_batch["task_names"] = task_names
        if self.task_to_id is not None:
            output_batch["task_ids"] = torch.LongTensor([self.task_to_id[task] for task in task_names])

        output_batch["labels_texts"] = labels
        return output_batch


class MMLUDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self, shuffle=False, workers=16):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self, shuffle=False, workers=16):
        return DataLoader(
            self.test_set,
            batch_size=self.config.predict_batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=self.collate_fn,
        )

    def __init__(self, config, data_dir=None, for_generation=False):
        super().__init__()

        self.data_dir = data_dir or config.data_dir
        self.config = config
        self.for_generation = for_generation
        self.tokenizer = get_tokenizer(config)
        self._setup()

    def get_dataset(self):
        import pkg_resources

        filename = pkg_resources.resource_filename(__name__, "../dataloader/mmlu_dataset.py")
        return load_dataset(filename, data_dir=self.data_dir)

    def setup(self, stage=None):
        pass

    def _setup(self):
        dataset = self.get_dataset()

        task_to_id = set(dataset["train"]["Task"])
        task_to_id = task_to_id.union(set(dataset["validation"]["Task"]))
        task_to_id = task_to_id.union(set(dataset["test"]["Task"]))
        self.task_to_id = {task: i for i, task in enumerate(task_to_id)}

        self.collate_fn = DataCollatorForMMLU(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family="seq2seq" if self.for_generation else self.config.model_family,
            task_to_id=self.task_to_id,
        )

        self.train_dataset = dataset["train"]
        self.test_set = self.dev_dataset = dataset["test"]

        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))


if __name__ == "__main__":
    from mttl.config import Config

    config = Config.parse()
    data_module = MMLUDataModule(config)
    data_module.setup()
