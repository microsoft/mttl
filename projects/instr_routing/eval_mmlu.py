from transformers import AutoModelForCausalLM
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.models.clm import CLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule


def eval_mmlu(
    config,
    model,
    data_dir=None,
    subsample=-1,
):
    from mttl.evaluators import MMLUEvaluator

    evaluator = MMLUEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
    )
    metrics = evaluator.evaluate(model, subsample=subsample)
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    from config import RoutingConfig
    from huggingface_hub import login
    
    
    

    config = RoutingConfig.parse(extra_kwargs={"eval_superni": True})

    config.model = "yahma/llama-7b-hf"
    config.load_in_8bit = 0 # True
    config.model_family = "gpt"
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 256
    config.max_output_length = 128
    config.model_modifier="softmoe"
    config.dataset = "alpaca"
    
    config.modify_modules=".*attn.*"        
    config.modify_layers="q_proj|v_proj|k_proj"   
    config.trainable_param_names=".*lora_[ab].*|.*selector.*"

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        load_in_8bit=config.load_in_8bit,
        device_map="auto"
    )
    model_class = CLM    
    dm = AlpacaDataModule(config)
    path_best_model = "/home/v-oostapenko/dev/mttl/tmp/instruction_learning/yahma_llama-7b-hf0qx192oq_None-val/loss=1.4099.ckpt"
    best_model = CLM.load_from_checkpoint(path_best_model, tokenizer=dm.tokenizer).cuda()
    # model = model_class(**vars(config), tokenizer=dm.tokenizer)
    # model.to("cuda")
      
    print(eval_mmlu(config, model, subsample=10))
    
    
    
    # if "HF_TOKEN" in os.environ:
    #     login(token=os.environ["HF_TOKEN"])

    # config = RoutingConfig.parse(extra_kwargs={"eval_superni": False})

    # config.model = "meta-llama/Llama-2-7b-hf"
    # config.load_in_8bit = True
    # config.model_family = "gpt"
    # config.data_dir = os.environ["MMLU_DATA_DIR"]
    # config.predict_batch_size = 2
    # config.max_input_length = 4096
    # config.max_output_length = 5

    # model = AutoModelForCausalLM.from_pretrained(
    #     config.model,
    #     load_in_8bit=config.load_in_8bit,
    #     device_map="auto"
    # )
    # print(eval_mmlu(config, model, subsample=10))

