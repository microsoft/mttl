import copy
import json
import torch   
import sys 
import numpy as np
import transformers          
from datasets import load_dataset           
from scipy.stats import entropy as calc_entropy
sys.path.append("/home/v-oostapenko/dev/mttl")   
from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
from typing import List, Sequence, Dict

import random

encoding_templates_w_input = [
    # input encoding template, output encoding template, weight
    ("{instruction}\n\n{input}\n\n", "{output}", 0.2),
    ("{instruction}\n{input}\n\n", "{output}", 0.1),
    ("{instruction}\n{input}\n", "{output}", 0.1),
    ("{instruction}\n\nInput: {input}\n\nOutput:", "{output}", 0.05),
    ("{instruction}\nInput: {input}\nOutput:", "{output}", 0.05),
    ("{instruction}\n{input}\n\nResponse:", "{output}", 0.05),
    ("{instruction}\n\nAdditional Context:\n{input}\n\nAnswer:", "{output}", 0.05),
    ("Task: {instruction}\nInput: {input}\nOutput:", "{output}", 0.05),
    ("Task: {instruction}\n\n{input}\n\n", "{output}", 0.05),
    ("Task: {instruction}\n\n{input}\n\nAnswer:", "{output}", 0.05),
    ("You need to complete the following task:\n\n{instruction}\n\n{input}\n\nAnswer:", "{output}", 0.05),
    ("{instruction}\n\nNow complete the following instance -\nInput: {input}\nOutput:", "{output}", 0.05),
    ("Instruction:{instruction}\n\nInput: {input}\n\n", "{output}", 0.05),
    ("Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:", "{output}", 0.1), # alpaca template
]

encoding_templates_wo_input = [
    ("{instruction}\n\n", "{output}", 0.2),
    ("{instruction}\n", "{output}", 0.1),
    ("{instruction}", "\n{output}", 0.1),
    ("{instruction} Output:", "{output}", 0.05),
    ("{instruction}\nResponse:", "{output}", 0.05),
    ("{instruction}\n\nAnswer:", "{output}", 0.05),
    ("Task: {instruction}\n\n", "{output}", 0.05),
    ("Instruction: {instruction}\n", "{output}", 0.05),
    ("Instruction: {instruction}\nOutput:", "{output}", 0.05),
    ("You need to complete the following task:\n\n{instruction}\n\n", "{output}", 0.05),
    ("Can you help with this?\n\n{instruction}\n", "{output}", 0.05),
    ("Plase answer the following request: {instruction}\nAnswer:", "{output}", 0.05),
    ("Tell me how would you respond to the following request.\n{instruction}\n", "{output}", 0.05),
    ("Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:", "{output}", 0.1), # alpaca template
]


def encode_instruction_example(instruction, input, output, random_template=True, eos_token=None):
    if random_template:
        if input is not None and input.strip() != "":
            # randomly choose a template with input
            prompt_template, completion_template, _ = random.choices(
                encoding_templates_w_input, weights=[w for _, _, w in encoding_templates_w_input]
            )[0]
            prompt = prompt_template.format(instruction=instruction.strip(), input=input.strip())
            completion = completion_template.format(output=output.strip())
        else:
            # randomly choose a template without input
            prompt_template, completion_template, _ = random.choices(
                encoding_templates_wo_input, weights=[w for _, _, w in encoding_templates_wo_input]
            )[0]
            prompt = prompt_template.format(instruction=instruction.strip())
            completion = completion_template.format(output=output.strip())
    else:
        if input is not None and input.strip() != "":
            prompt = instruction.strip() + "\n\n" + input.strip() + "\n\n"
            completion = output.strip()
        else:
            prompt = instruction.strip() + "\n\n"
            completion = output.strip()

    data = {
        "prompt": prompt,
        "completion": completion + eos_token if eos_token else completion,
    }
    return data


def encode_few_shot_example(instruction, examplars, input, output, eos_token=None):
    prompt = instruction.strip() + "\n\n"
    for examplar in examplars:
        prompt += "Input:\n" + examplar["input"].strip() + "\n"
        prompt += "Output:\n" + examplar["output"].strip() + "\n\n"

    prompt += "Input:\n" + input.strip() + "\n"
    prompt += "Output:\n"

    data = {
        "prompt": prompt,
        "completion": output.strip() + eos_token if eos_token else output.strip(),
    }
    return data


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text,
                return_tensors='pt', 
                padding="max_length",  
                max_length=max_seq_length,
                truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(   
                    _concat_messages(messages[:message_idx]), 
                                     return_tensors='pt', 
                                     max_length=max_seq_length, 
                                    #  padding="max_length", 
                                     truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                # padding="max_length",  
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]  
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break
    # output_text = tokenizer.decode(input_ids[0])
    # instruction = output_text.split("<|assistant|>")[0].strip()
    # instruction.replace("<|system|>", "").replace("<|user|>", "")
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),  
        'attention_mask': attention_mask.flatten(),
        'input_text': example_text,
        'instruction': example_text, 
        'hash_text': str(example['messages'])
    }


class HumanMixDataset(torch.utils.data.dataset.Dataset):     
    def __init__(self, tokenizer, 
                 max_input_length,            
                 max_output_length, data_dir=None,               
                 train_on_inputs=False, dst_path=None, idxs=None, cluster_info=None, predict_cluster=None, loss_for_keywords=True):
        super().__init__()
        self.dst_path=dst_path   
        self.predict_cluster = predict_cluster  
        self.loss_for_keywords = loss_for_keywords
        if predict_cluster is not None:
            assert predict_cluster in ["topic_set", "skill_set"]
        self.cluster_info = cluster_info
        self.train_on_inputs = train_on_inputs
        datasets={
            "flan_v1": "/home/v-oostapenko/dev/open-instruct/data/processed/flan_v2/flan_v2_data.jsonl",
            # "CoT": "/home/v-oostapenko/dev/open-instruct/data/processed/cot/cot_data.json",  
            # "Dolly": "/home/v-oostapenko/dev/open-instruct/data/processed/dolly/dolly_data.json",
            # "oasst1": "/home/v-oostapenko/dev/open-instruct/data/processed/oasst1/oasst1_data.json",
        }
        # import json     
        # from datasets import Dataset
        # import pandas as pd  
        self.dataset = load_dataset("json", data_files=datasets.values())["train"]
        if idxs is not None:     
            self.dataset = self.dataset.select(idxs) 
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
        if "prompt" in entry.keys() and "completion" in entry.keys():
            raise NotImplementedError("This is not implemented yet.")
        elif "messages" in entry.keys(): 
            enc_input = encode_with_messages_format(entry, self.tokenizer, self.max_input_length)
        else:
            raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
                        
        hash = hash_example(enc_input["hash_text"])
        
        task_id = -1   
        if self.cluster_info is not None:
            task_id = self.cluster_info.get_distances(hash)
            # for low entropy argmax
            entr = calc_entropy(task_id, axis=-1)/ np.log2(len(task_id))
            if entr>0.4: # what was used in gen_si_sets to generate datasets and clusterings
                task_id = -2
            else:
                task_id = torch.tensor(task_id).argmax().item() # its probs actually, not distances TODO: deal with this ambiguity
            # for high entropy -2
                   
        ex_info = ExampleInfo( 
            enc_input["input_ids"],
            enc_input["labels"],
            task_id,
            hash,
            example_id=key,
            input_text=(enc_input["input_text"]),
            instruction_hash=None
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions

if __name__ == "__main__":         
    from transformers import LlamaForCausalLM, LlamaTokenizer           
    tok_model = "yahma/llama-7b-hf"   
    tokenizer = LlamaTokenizer.from_pretrained(tok_model, add_eos_token=True)
    max_input_length = 512
    max_output_length = 128
    dataset = HumanMixDataset(tokenizer, max_input_length, max_output_length)
    for i in range(10):
        print(dataset[i].input_text)