import copy
import torch    
import json
import io      
import transformers
from tqdm import tqdm
from datasets import load_dataset

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
from typing import List, Sequence, Dict
import logging
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\n### Response:"
    ),
    "prompt_no_input": (
        "{instruction}\n\n### Response:"
    ),
}


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return 

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _tokenize_fn(strings: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = tokenizer(
            strings,     
            truncation=True,
            padding="max_length", 
            max_length=tokenizer.model_max_length,
            return_tensors="pt")
    input_ids = labels = tokenized_list.input_ids[0]
    input_ids_lens = labels_lens = torch.logical_and(input_ids.ne(tokenizer.pad_token_id),input_ids.ne(tokenizer.eos_token_id)).sum().item()
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(   
    source: Sequence[str],
    target: Sequence[str],      
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    example = source + target # [s + t for s, t in zip(sources, targets)]
    example_tokenized = _tokenize_fn(example, tokenizer) #for strings in (examples, sources)]
    sources_tokenized = _tokenize_fn(source, tokenizer) 
    input_ids = example_tokenized["input_ids"]
    label = copy.deepcopy(input_ids)        
    # for label, source_len in tqdm(zip(labels, sources_tokenized["input_ids_lens"]), total=len(labels)):
    source_len = sources_tokenized["input_ids_lens"]
    label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=label)

           
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict =  load_dataset('WizardLM/WizardLM_evol_instruct_V2_196k')['train'] #"json", data_files=data_path+"/WizardLM_evol_instruct_V2_143k.json")["train"] #'WizardLM/WizardLM_evol_instruct_V2_196k')['train']
        #jload(data_path)
        self.tokenizer = tokenizer
        logging.warning("Formatting inputs...")    
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        self.sources = [
            prompt_input.format_map({'instruction':example['conversations'][0]["value"]}) if example['conversations'][0].get("input", "") != "" else prompt_no_input.format_map({'instruction':example['conversations'][0]["value"]})
            for example in list_data_dict
        ]  
        self.targets = [f"{example['conversations'][1]['value']}{tokenizer.eos_token}" for example in list_data_dict]


        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(self.sources, self.targets, tokenizer)

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]

    def __len__(self):      
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        enc_input = self.sources[i]        
        input_hash = hash_example(self.sources[i]+self.targets[i])
        instruction_hash = hash_example(self.sources[i])
        processed=preprocess(self.sources[i], self.targets[i], self.tokenizer)
        
        ex_info = ExampleInfo(
            processed["input_ids"],
            processed["labels"],
            -1,
            input_hash,
            example_id=i,
            input_text=(enc_input), 
            instruction_hash=instruction_hash,
        )
        return ex_info

class SupervisedComplexDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedComplexDataset, self).__init__()
        logging.warning("Loading data...")
        #list_data_dict = utils.jload(data_path)

        list_data_dict = []

        with open(data_path) as f:
            for line in f:    
                cur_obj = json.loads(line)
                cur_obj = json.loads(cur_obj)
                list_data_dict.append(cur_obj)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            (example['instruction'].strip()+'\n\n') for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

       
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])