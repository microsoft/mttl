import torch 
import sys
import os
import copy
from transformers import LlamaForCausalLM
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.modifiers import modify_transformer
from models.clm import CLM, prepare_model_for_kbit_training
from projects.instr_routing.finetune_llama import RoutingConfig
from mttl.datamodule.alpaca_data_module import AlpacaDataModule



config = RoutingConfig()
config.model="yahma/llama-7b-hf"
config.model_family = "gpt"
config.model_modifier = "vsmear"
config.modify_modules=".*attn.*"       
config.modify_layers="q_proj|v_proj|k_proj"   



dm = AlpacaDataModule(config)
model_object = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path="yahma/llama-7b-hf",
    load_in_8bit=1,
    torch_dtype=torch.float32,
    device_map="auto",
)
model_object = prepare_model_for_kbit_training(model_object)
model_object = modify_transformer(model_object, config)

module = CLM(tokenizer=dm.tokenizer, model_object=model_object, **vars(config)).cuda()
batch = ["tes test tes","tes test tes","tes test tes"]
input_ids = module.tokenizer(batch, return_tensors="pt", padding=True).input_ids
labels = copy.deepcopy(input_ids)
labels[:,-1]=-100
batch={"input_ids": input_ids.cuda(), "labels": labels.cuda(), "task_ids": torch.tensor([0]).cuda()}
loss, aux_loss = module.forward(batch)
print(aux_loss)