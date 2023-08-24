import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from typing import List

from mttl.utils import get_ni_tasks_from_file, trim_batch, hash_example
from mttl.datamodule.utils import get_tokenizer
from mttl.datamodule import IndexConcatDataset
from mttl.datamodule.collators import DefaultCollator
from mttl.dataloader.ni_dataset_readers import NIDatasetReader


class NIDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def __init__(self, config):
        super().__init__()

        self.config = config

        if not self.config.finetune_task_name:
            self.tasks, self.task2id = get_ni_tasks_from_file(
                config.custom_tasks_splits
            )
        else:
            self.task_name = self.config.finetune_task_name
            self.task2id = {self.task_name: 0}
            self.tasks = [self.task_name]

        self.dataset_reader: NIDatasetReader = None
        self.id2task = dict((k, v) for v, k in self.task2id.items())

        self.tokenizer = get_tokenizer(config)
        self.collate_fn = DefaultCollator(
            tokenizer=self.tokenizer,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=config.model_family,
        )
        if config.embeddings_path:
            self.task_embed_path = config.embeddings_path
        else:
            self.task_embed_path = None

    @property
    def full_dataset(self):
        return torch.utils.data.dataset.ConcatDataset(
            [self.train_dataset, self.val_dataset, self.test_dataset]
        )

    @property
    def all_instructions(self):
        """Return all task instructions used in the dataset.
        """
        return self.dataset_reader.read_all_instructions()

    @property
    def dataset_name(self):
        return hash_example("-".join(self.tasks))

    def setup(self, stage="fit", val_examples_per_task=None, test_examples_per_task=None):
        self.dataset_reader = NIDatasetReader(
            self.config.data_dir,
            self.tokenizer,
            tasks=self.tasks,
            task2id=self.task2id,
            task_embed_path=self.task_embed_path,
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            use_task_descriptions=self.config.use_task_descriptions,
            num_positive_examples=self.config.num_pos_examples,
            val_examples_per_task=val_examples_per_task,
            test_examples_per_task=test_examples_per_task,
        )

        print("Training on the following tasks: {}".format(self.tasks))

        self.train_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("train")
        )
        self.val_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("dev")
        )
        self.test_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("test")
        )

        print("Training examples:", len(self.train_dataset))
        print("Validation examples:", len(self.val_dataset))
        print("Test examples:", len(self.test_dataset))


class NIPretrainDataModule(NIDataModule):
    pass


class NIZeroShotDataModule(NIDataModule):
    pass


class NIFinetuneDataModule(NIDataModule):
    pass
