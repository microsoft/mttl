import torch   
import sys 
import os        
from datasets import load_dataset           

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example
                 
def encode_with_messages_format(example):
     
    message_text = ""
    intruction = "<|user|>\n" + example["user"].strip() + "\n"
    intruction += "<|assistant|>\n"  
    output = example["assistant"].strip()
    message_text += intruction
    message_text += output
    message_text = message_text.strip()
     
    return {           
        'input_text': intruction,
        "output_text": output,   
        'full_text': message_text, 
        'hash_text':message_text
    }


class Flan100KDataset(torch.utils.data.dataset.Dataset):     
    def __init__(self,    
                 data_dir=None,
                 idxs=None,     
                 loss_for_keywords=True):
        super().__init__()
        self.loss_for_keywords = loss_for_keywords
            
        if os.getenv("AP_DATA_DIR") is not None:
            data_dir = os.getenv("AP_DATA_DIR")  
        
        self.dataset = load_dataset("ostapeno/flanv2_100k_2", cache_dir=data_dir)["train"]
        if idxs is not None:     
            self.dataset = self.dataset.select(idxs) 

    def __len__(self):     
        return len(self.dataset)
    
    def __getitem__(self, key):
        entry = self.dataset[key]           
        enc_input = encode_with_messages_format(entry)
                        
        hash = hash_example(enc_input["hash_text"])
        
        task_id = -1  
        ex_info = ExampleInfo( 
            enc_input["input_text"],
            enc_input["output_text"],
            task_id,
            hash,
            example_id=key,
            input_text=(enc_input["full_text"]),
            instruction_hash=None
        )    
        return ex_info

if __name__ == "__main__":        
    dataset = Flan100KDataset()
    for i in range(10):
        print(dataset[i].input_text)