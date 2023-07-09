import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mttl.datamodule.ni_data_module import CollateWrapperFn, CollateWrapperFnCLM
from mttl.dataloader.wizard_dataset_readers import WizardDataset
from transformers import LlamaTokenizer


class WizardDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFnCLM(self.pad_token_id),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFnCLM(self.pad_token_id),
        )

    def test_dataloader(self):
        return DataLoader(
            self.dev_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFnCLM(self.pad_token_id),
        )

    @property
    def all_instructions(self):
        return self.dataset.read_all_instructions()

    def __init__(self, config):
        super().__init__()

        self.config = config

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "/home/v-oostapenko/llms/", #config.model,
        #     model_max_length=config.max_input_length
        # )

        tok_model = config.model if config.model is not None else "yahma/llama-7b-hf"
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tok_model, add_eos_token=True
        )  # tloen does not add eos token
        # self.tokenizer.pad_token_id =
        self.tokenizer.pad_token_id = 0

        if self.config.padding_side == "left":
            self.tokenizer.padding_side = (
                "left"  # Allow batched inference, used by tloen also in training
            )
        self.pad_token_id = self.tokenizer.pad_token_id

        self.task2id = {"alpaca_full": 0}

    def get_dataset(self):
        return WizardDataset(
            self.tokenizer,
            self.config.max_input_length,
            self.config.max_output_length,
            self.config.train_dir,
            self.config.train_on_inputs,
        )

    def setup(self, stage=None):
        dataset = self.get_dataset()

        # always use the same split for the dataset
        rng = torch.Generator().manual_seed(1234)

        n_tr_samples = int(len(dataset) * 0.97)  # len(dataset) -
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


class WizardPretrainDataModule(WizardDataModule):
    pass


class WizardFinetuneDataModule(WizardDataModule):
    pass
