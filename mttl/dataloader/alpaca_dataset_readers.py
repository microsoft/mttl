import copy
import json
import torch   
import os
import numpy as np
import transformers
from datasets import load_dataset
from scipy.stats import entropy as calc_entropy

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
from typing import List, Sequence, Dict

class AlpacaTemplateForHash(object): # dont change it to keep compatibility with old clusterings etc., previously generated hashes
    @classmethod        
    def apply(self, dict_values):   
        instruction, input, output = dict_values["instruction"], dict_values["input"], dict_values["output"]
        if len(input)>0:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Input:{input}\
            \n### Response: {output}"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Response: {output}"

class AlpacaTemplate(object):
    @classmethod                            
    def apply(self, dict_values, topics_str=None):      
        instruction, input, output = dict_values["instruction"], dict_values["input"], dict_values["output"]
        if len(input)>0:
            instr = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Input: {input}"
        else:
            instr = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}"
        # if topics_str is not None:  
        #         instr += f"\n###Response: [ {topics_str} ] {output}"
        # else:
        instr += f"\n### Response: {output}"
        return instr

class AlpacaTemplateSource(object):
    @classmethod
    def apply(self, dict_values):
        instruction, input, output = dict_values["instruction"], dict_values["input"], dict_values["output"]
        if len(input)>0:
            return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Input: {input}\
            \n### Response:"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {instruction}\
            \n### Response:"


class AlpacaDataset(torch.utils.data.dataset.Dataset):     
    def __init__(self, tokenizer, max_input_length,            
                 max_output_length, data_dir,               
                 train_on_inputs=False, dst_path=None, idxs=None, cluster_info=None, predict_cluster=None, loss_for_keywords=True):
        super().__init__()
        self.dst_path=dst_path   
        self.predict_cluster = predict_cluster
        self.loss_for_keywords = loss_for_keywords
        if predict_cluster is not None:
            assert predict_cluster in ["topic_set", "skill_set"]
        self.cluster_info = cluster_info
        self.train_on_inputs = train_on_inputs
        # load the data 
        if os.getenv("AP_DATA_DIR") is not None:
            data_dir = os.getenv("AP_DATA_DIR")
        if dst_path is None:       
            self.dataset = load_dataset("yahma/alpaca-cleaned", cache_dir=data_dir)["train"]
            if idxs is not None:       
                self.dataset = self.dataset.select(idxs)
        else: 
            import json     
            from datasets import Dataset
            import pandas as pd
            with open(dst_path, "r") as f:
                new_dataset = json.load(f)
            for ex in new_dataset:
                if "topic_set" in ex:
                    ex["topic_set"] = str(ex["topic_set"])
                if "skill_set" in ex:
                    ex["skill_set"] = str(ex["skill_set"])
            df = pd.DataFrame(new_dataset)
            if idxs is not None:
                df = df.iloc[idxs]
            # transform into dataset
            self.dataset = Dataset.from_pandas(df)
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
        # TODO: check with AS if this is OOP approved
        
        enc_input_for_hash = AlpacaTemplateForHash.apply(entry)
        input_hash = hash_example(enc_input_for_hash) 
        instruction_hash = hash_example(entry["instruction"])
        topics_str = None    
        
        enc_input = AlpacaTemplate.apply(entry)
        source = AlpacaTemplateSource.apply(entry)
        
        if self.predict_cluster is not None and self.predict_cluster in entry:
            str_dict = entry[self.predict_cluster].replace("'", "\"")
            topic_set = json.loads(str_dict)
            topics_str = " ".join([k for k,v in topic_set.items()])             
            if self.loss_for_keywords:
                entry["output"] = f"[keywords: {topics_str} ] {entry['output']}"
            else:
                source += f" [keywords: {topics_str} ]"
        # assert source + entry["output"] == enc_input
        # dec_input = entry["output"]
        task_id = -1
        if self.cluster_info is not None:
            task_id = self.cluster_info.get_distances(input_hash)
            # for low entropy argmax
            entr = calc_entropy(task_id, axis=-1)/ np.log2(len(task_id))
            if entr>0.4: # what was used in gen_si_sets to generate datasets and clusterings
                task_id = -2
            else:
                task_id = torch.tensor(task_id).argmax().item() # its probs actually, not distances TODO: deal with this ambiguity
            # for high entropy -2
        if self.train_on_inputs:
        # next we tokenize      
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
            task_id, # task id
            input_hash,
            example_id=key,
            input_text=(enc_input),
            instruction_hash=instruction_hash)
            return ex_info
        tok_input = self.preprocess(source, entry["output"])        
        ex_info = ExampleInfo(
            tok_input["input_ids"],
            tok_input["labels"],
            task_id,
            input_hash,
            example_id=key,
            input_text=(enc_input),
            instruction_hash=instruction_hash,
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions
