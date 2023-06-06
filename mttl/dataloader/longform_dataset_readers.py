import copy
import torch    
import transformers
from datasets import load_dataset

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
from typing import List, Sequence, Dict


class LongFormTemplate(object):
    @classmethod
    def apply(self, dict_values):                             
        instruction, output = dict_values["input"], dict_values["output"]
        return f"{instruction} [EOI]"


class LongFormDataset(torch.utils.data.dataset.Dataset):     
    def __init__(self, tokenizer, max_input_length, max_output_length, data_dir, train_on_inputs=False, split="train"):
        super().__init__()
        self.train_on_inputs = train_on_inputs
        # load the data 
        self.dataset = load_dataset("akoksal/LongForm", cache_dir=data_dir)[split]
        # each entry is "instruction", "input", "output" dictionary

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):     
        return len(self.dataset)

    def _tokenize_fn(self, string: str) -> Dict:
        """Tokenize a list of strings."""
        tokenized =self.tokenizer(
                string,     
                truncation=True,
                padding="max_length", 
                max_length=self.max_input_length,
                return_tensors="pt"
            )
        input_ids = labels = tokenized.input_ids[0] 
        # input_ids_lens = labels_lens = tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
        # input_ids_lens = labels_lens = tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
        input_ids_lens = labels_lens = torch.logical_and(tokenized.input_ids.ne(self.tokenizer.pad_token_id),tokenized.input_ids.ne(self.tokenizer.eos_token_id)).sum().item()
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )
    
    def preprocess(self,
        source: str,       
        target: str) -> Dict:
        IGNORE_INDEX=-100
        """Preprocess the data by tokenizing."""    
        # _tokenize_fn = lambda x: self.tokenizer(x,
        #     truncation=True,
        #     padding="max_length",
        #     max_length=self.max_input_length,
        #     return_tensors="pt"
        #     )
        example = source + target #[s + t for s, t in zip(sources, targets)]
        example_tokenized = self._tokenize_fn(example)
        sources_tokenized = self._tokenize_fn(source) #[_tokenize_fn(strings) for strings in (examples, sources)]
        input_ids = example_tokenized["input_ids"]
        label = copy.deepcopy(input_ids)
        # for label, source_len in zip(label, sources_tokenized["input_ids_lens"]):
        label[:sources_tokenized["input_ids_lens"]] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=label)
    
    def __getitem__(self, key):
        entry = self.dataset[key]
        # really basic template for now
        enc_input = LongFormTemplate.apply(entry)
        input_hash = hash_example(enc_input)
        instruction_hash = input_hash #hash_example(entry["instruction"])        
        source = enc_input #AlpacaTemplateSource.apply(entry)
        # dec_input = entry["output"]
        if self.train_on_inputs:
            # next we tokenize      
            raise NotImplementedError("train on inputs not implemented")
            tok_input = self.tokenizer(
                enc_input, 
                truncation=True,
                padding="max_length",
                max_length=self.max_input_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            ex_info = ExampleInfo(
            tok_input,
            tok_input,
            -1,
            input_hash,
            example_id=key,
            input_text=(enc_input),
            instruction_hash=instruction_hash)
            return ex_info
        tok_input = self.preprocess(source, entry["output"])        
        ex_info = ExampleInfo(
            tok_input["input_ids"],
            tok_input["labels"],
            -1,
            input_hash,
            example_id=key,
            input_text=(enc_input+entry["output"]),
            instruction_hash=instruction_hash,
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions
