from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mttl.datamodule.ni_data_module import CollateWrapperFn
from mttl.dataloader.alpaca_dataset_readers import AlpacaDataset


class AlpacaDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
        )

    def val_dataloader(self):
        print('Using Train set as valid')
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
        )

    def test_dataloader(self):
        raise NotImplementedError

    @property
    def all_instructions(self):
        return self.dataset.read_all_instructions()

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, model_max_length=config.max_input_length
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        self.task2id = {}

    def setup(self, stage=None):
        self.train_dataset = AlpacaDataset(
            self.tokenizer, self.config.max_input_length, self.config.max_output_length
        )
        # self.val_dataset = IndexConcatDataset(
        #     self.dataset_reader.read_orig_datasets("dev")
        # )

        print("Training steps:", len(self.train_dataloader()))
        # print("Validation steps:", len(self.val_dataloader()))


class AlpacaPretrainDataModule(AlpacaDataModule):
    pass


class AlpacaFinetuneDataModule(AlpacaDataModule):
    pass
