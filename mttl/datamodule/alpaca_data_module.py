import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import List
from mttl.dataloader.data_utils import ExampleInfo
from mttl.dataloader.alpaca_data_readers import AlpacaDataset
from transformers import AutoTokenizer


class CollateWrapperFn:
    def __init__(
        self,
        pad_token_id,
        pad_left=False,
    ):
        self.pad_token_id = pad_token_id
        self.pad_left = pad_left

    def __call__(self, batch: List[ExampleInfo]):
        input_ids = [b.input_ids for b in batch]
        target_ids = [b.target_ids for b in batch]
        hashes = [b.hash for b in batch]
        task_ids = [b.task_id for b in batch]
        instruction_hashes = [b.instruction_hash for b in batch]

        task_ids = torch.LongTensor(task_ids)
        max_len_s = max(len(s) for s in input_ids)
        max_len_t = max(len(s) for s in target_ids)

        def pad_right(x, pad_id, max_len):
            # pad right based on the longest sequence
            n = max_len - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

        def pad_left(x, pad_id, max_len):
            # pad left based on the longest sequence
            n = max_len - len(x)
            return torch.cat((torch.full((n,), pad_id, dtype=x.dtype), x))

        if not self.pad_left:
            input_ids = torch.stack([pad_right(x, pad_id=0, max_len=max_len_s) for x in input_ids])
            target_ids = torch.stack([pad_right(x, pad_id=-100, max_len=max_len_t) for x in target_ids])
        else:
            input_ids = torch.stack([pad_left(x, pad_id=0, max_len=max_len_s) for x in input_ids])
            target_ids = torch.stack([pad_left(x, pad_id=-100, max_len=max_len_t) for x in target_ids])

        output_batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "task_ids": task_ids,
            "hashes": hashes,
            "instruction_hashes": instruction_hashes,
        }
        return output_batch


class AlpacaDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id, self.pad_left),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id, self.pad_left),
        )

    def test_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id, self.pad_left),
        )

    @property
    def all_instructions(self):
        return self.dataset.read_all_instructions()

    def __init__(self, config, generation_mode=False):
        super().__init__()

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, add_eos_token=True
        )  # tloen does not add eos token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = 0

        self.pad_left = generation_mode
        self.generation_mode = generation_mode
        self.pad_token_id = self.tokenizer.pad_token_id
        self.task2id = {"alpaca_full": 0}

    def get_dataset(self):
        return AlpacaDataset(
            self.tokenizer,
            self.config.max_length,
            self.config.train_dir,
            prepare_for_generation=self.generation_mode,
        )

    def setup(self, stage=None):
        dataset = self.get_dataset()

        # always use the same split for the dataset
        rng = torch.Generator().manual_seed(1234)

        n_tr_samples = int(len(dataset) * 0.95)

        self.train_dataset, self.dev_dataset = torch.utils.data.random_split(
            dataset,
            [
                n_tr_samples,
                len(dataset) - n_tr_samples,
            ],
            generator=rng,
        )

        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))


class AlpacaPretrainDataModule(AlpacaDataModule):
    pass


class AlpacaFinetuneDataModule(AlpacaDataModule):
    pass
