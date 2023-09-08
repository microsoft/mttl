

from transformers import AutoModelForCausalLM
import sys
import os  
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
import os
import torch
torch.set_float32_matmul_precision('high')


def eval_ni(
    config,
    model,
    nshot=2,
    data_dir=None,
    subsample=-1,
    max_input_length=None,
):
    from mttl.evaluators import NIEvaluator

    ni_evaluator = NIEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
        num_pos_examples=nshot,
        max_input_length=max_input_length
    )
    metrics = ni_evaluator.evaluate(model, subsample=subsample)
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__": 
    from huggingface_hub import login  
    # check with loading      
    config = RoutingConfig.parse(c="/home/v-oostapenko/dev/mttl/projects/instr_routing/configs/alpaca/llama1_7b_vsmear.json")
    login(token=os.environ["HF_TOKEN"])
    dm = AlpacaDataModule(config)   
    path_best_model = "/home/v-oostapenko/dev/amlt/routing/alpaca_dense_r4/yahma_llama-7b-hf3op8p057_alpaca_dense_r4-val/loss=0.8786.ckpt"
    best_model = CLM.load_from_checkpoint(path_best_model, tokenizer=dm.tokenizer).cuda()
    config_best = best_model.hparams
    config_best.data_dir = os.environ["NI_DATA_DIR"]      
    config_best.output_dir = config.output_dir
    config_best.predict_batch_size=5 
    print(eval_ni(config_best, best_model, nshot=0, subsample=-1, max_input_length=-1))
    
    
    # login(token=os.environ["HF_TOKEN"])
    # config = RoutingConfig.parse(extra_kwargs={"eval_superni": True})
    # config.model = "meta-llama/Llama-2-7b-hf"
    # config.load_in_8bit = True
    # config.model_family = "gpt"
    # config.data_dir = os.environ["NI_DATA_DIR"]
    # config.predict_batch_size = 2
    # config.max_input_length = 4096
    # config.max_output_length = 128

    # model = AutoModelForCausalLM.from_pretrained(
    #     config.model,
    #     load_in_8bit=config.load_in_8bit,
    #     device_map="auto"
    # ) 
    # print(eval_ni(config, model, nshot=0, subsample=-1))
