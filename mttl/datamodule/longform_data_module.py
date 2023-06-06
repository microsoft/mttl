import torch      
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from mttl.datamodule.ni_data_module import CollateWrapperFn, CollateWrapperFnCLM
from mttl.dataloader.longform_dataset_readers import LongFormDataset
from transformers import LlamaTokenizer

class LongFormDataModule(LightningDataModule):
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
            self.test_dataset,
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
        self.tokenizer = LlamaTokenizer.from_pretrained(tok_model, add_eos_token=True) # tloen does not add eos token
        # self.tokenizer.pad_token_id = 
        self.tokenizer.pad_token_id = 0
        #Add [EOI] token to the tokenizer                  
        # self.tokenizer.add_tokens("[EOI]", special_tokens=True)
        # get [EOI] token id
        # self.eoi_token_id = self.tokenizer.encode("[EOI]")[0]
        new_tokens = ["[EOI]"]
        new_tokens = set(new_tokens) - set(self.tokenizer.get_vocab().keys())
        self.tokenizer.add_tokens(list(new_tokens))
        
        if self.config.padding_side == "left":
            self.tokenizer.padding_side = "left"  # Allow batched inference, used by tloen also in training
        self.pad_token_id = self.tokenizer.pad_token_id

        self.task2id = {'alpaca_full':0}

    def get_dataset(self, split="train"):
        return LongFormDataset(
            
            self.tokenizer, self.config.max_input_length, self.config.max_output_length, self.config.train_dir, self.config.train_on_inputs, split=split
        )
    
    def setup(self, stage=None):           
        self.train_dataset = self.get_dataset(split="train")
        self.dev_dataset = self.get_dataset(split="validation")
        self.test_dataset = self.get_dataset(split="test")
        # )      
        # max sample length
        # a=[len(s.input_text.split(" ")) for s in self.train_dataset]
        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))


# class AlpacaPretrainDataModule(AlpacaDataModule):
#     pass


# class AlpacaFinetuneDataModule(AlpacaDataModule):
#     pass
