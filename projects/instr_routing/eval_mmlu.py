from transformers import AutoModelForCausalLM
import os
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
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
    from huggingface_hub import login    

    config = RoutingConfig.parse(c="/home/v-oostapenko/dev/mttl/projects/instr_routing/configs/alpaca/llama1_7b_vsmear.json")

    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    dm = AlpacaDataModule(config)
    path_best_model = "/home/v-oostapenko/dev/mttl/tmp/instruction_learning/yahma_llama-7b-hf0qx192oq_None-val/loss=1.4099.ckpt"
    best_model = CLM.load_from_checkpoint(path_best_model, tokenizer=dm.tokenizer).cuda()
    result = eval_mmlu(config, best_model) #, subsample=20)
    print(result)
    
    
    
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

