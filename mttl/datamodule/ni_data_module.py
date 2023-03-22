import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import List

from mttl.utils import get_ni_tasks_from_file, trim_batch, hash_example
from mttl.datamodule import IndexConcatDataset
from mttl.dataloader.data_utils import ExampleInfo
from mttl.dataloader.ni_dataset_readers import NIDatasetReader


class CollateWrapperFn:
    def __init__(
        self,
        pad_token_id,
    ):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[ExampleInfo]):
        input_ids = [b.input_ids for b in batch]
        target_ids = [b.target_ids for b in batch]
        hashes = [b.hash for b in batch]
        task_ids = [b.task_id for b in batch]
        instruction_hashes = [b.instruction_hash for b in batch]

        task_ids = torch.LongTensor(task_ids)
        input_ids = trim_batch(torch.stack(input_ids, 0), self.pad_token_id)
        target_ids = trim_batch(torch.stack(target_ids, 0), self.pad_token_id)

        output_batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "task_ids": task_ids,
            "hashes": hashes,
            "instruction_hashes": instruction_hashes,
        }
        return output_batch


class NIDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
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

        self.id2task = dict((k, v) for v, k in self.task2id.items())
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.tokenizer.model_max_length = config.max_input_length
        self.pad_token_id = self.tokenizer.pad_token_id

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

    def setup(self, stage=None):
        self.dataset_reader = NIDatasetReader(
            self.config.train_dir,
            self.tokenizer,
            tasks=self.tasks,
            task2id=self.task2id,
            task_embed_path=self.task_embed_path,
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            use_task_descriptions=self.config.use_task_descriptions,
            num_positive_examples=self.config.num_pos_examples,
        )

        print("Training on the following tasks: {}".format(self.tasks))

        train_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("train")
        )
        self.train_dataset = train_dataset
        self.val_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("dev")
        )
        self.test_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("test")
        )

        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))
        print("Test steps:", len(self.test_dataloader()))


class NIPretrainDataModule(NIDataModule):
    pass


class NIZeroShotDataModule(NIDataModule):
    pass


class NIFinetuneDataModule(NIDataModule):
    pass
