import copy
import json
import torch   
import transformers
from datasets import load_dataset

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
from typing import List, Sequence, Dict



INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42

# This is a training prompt that does not contain an input string.  The instruction by itself has enough information
# to respond.  For example, the instruction might ask for the year a historic figure was born.
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

PROMPT_FOR_GENERATION_WITH_INPUT_FORMAT = """{intro}


{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
""".format(
    intro=INTRO_BLURB,  
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
)



class DollyTemplate:
    @classmethod
    def apply(self, rec):       
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
        return rec["text"]

class DollyTemplateSource:
    @classmethod
    def apply(self, rec):
        
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")
        if context:
            rec["text"] = PROMPT_FOR_GENERATION_FORMAT.format(instruction=instruction)
        else:
            rec["text"] = PROMPT_FOR_GENERATION_WITH_INPUT_FORMAT.format(instruction=instruction, input=context)
        return rec["text"]


class DB_DollyDataset(torch.utils.data.dataset.Dataset):     
    def __init__(self, tokenizer, max_input_length,            
                 max_output_length, data_dir,               
                 train_on_inputs=False, dst_path=None, idxs=None, cluster_info=None, predict_cluster=None, loss_for_keywords=True):
        super().__init__()
        self.dst_path=dst_path   
        self.predict_cluster = predict_cluster  
        self.loss_for_keywords = loss_for_keywords
        # if predict_cluster is not None:
        #     assert predict_cluster in ["topic_set", "skill_set"]
        self.cluster_info = cluster_info
        self.train_on_inputs = train_on_inputs
        # load the data 
        self.dataset = load_dataset("databricks/databricks-dolly-15k", cache_dir=data_dir)["train"] 
        if idxs is not None:
            self.dataset = self.dataset.select(idxs)
        # ['brainstorming', 'classification', 'closed_qa', 'creative_writing', 'general_qa', 'information_extraction', 'open_qa', 'summarization']
        # each entry is "instruction", "input", "output" dictionary
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def categories(self):
        return self.dataset["categories"]

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
        
        enc_input_for_hash = DollyTemplate.apply(entry)
        input_hash = hash_example(enc_input_for_hash) 
        instruction_hash = hash_example(entry["instruction"])
        topics_str = None    
        
        enc_input = DollyTemplate.apply(entry)
        source = DollyTemplateSource.apply(entry)
        
        if self.predict_cluster is not None and self.predict_cluster in entry:
            str_dict = entry[self.predict_cluster].replace("'", "\"")
            topic_set = json.loads(str_dict)
            topics_str = " ".join([k for k,v in topic_set.items()])             
            if self.loss_for_keywords:
                entry["output"] = f"[keywords: {topics_str} ] {entry['response']}"
            else:
                source += f" [keywords: {topics_str} ]"
        # assert source + entry["output"] == enc_input
        # dec_input = entry["output"]
        task_id = -1
        if self.cluster_info is not None:
            task_id = self.cluster_info.get_distances(input_hash)
            task_id = torch.tensor(task_id).argmax().item() # its probs actually, not distances TODO: deal with this ambiguity
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
            instruction_hash=instruction_hash,
            category = entry["category"],)
            return ex_info
        tok_input = self.preprocess(source, entry["response"])        
        ex_info = ExampleInfo(
            tok_input["input_ids"],
            tok_input["labels"],
            task_id,
            input_hash,
            example_id=key,
            input_text=(enc_input),
            instruction_hash=instruction_hash,
            category = entry["category"]
        )
        return ex_info

    def read_all_instructions(self):
        """Read all instructions from the dataset."""
        all_instructions = []
        for data in self.dataset:
            all_instructions.append(data["instruction"])
        return all_instructions
