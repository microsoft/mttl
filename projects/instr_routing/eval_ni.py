
import sys
import os  
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from transformers import AutoModelForCausalLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
import os
import torch


def eval_ni(
    config,
    model,
    nshot=2,
    data_dir=None,
    eval_batches=-1,
):
    from mttl.evaluators import NIEvaluator

    ni_evaluator = NIEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
        num_pos_examples=nshot
    )
    metrics = ni_evaluator.evaluate(model, eval_batches=eval_batches)
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    from huggingface_hub import login

    config = RoutingConfig.parse()

    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])

    config = RoutingConfig.parse(extra_kwargs={"eval_superni": True})

    config.model = "yahma/llama-7b-hf"
    config.load_in_8bit = 0 # True
    config.model_family = "gpt"
    config.data_dir = os.environ["NI_DATA_DIR"]
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
      
    print(eval_ni(config, model, nshot=0, eval_batches=50))
