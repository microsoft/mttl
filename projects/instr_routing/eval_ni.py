

from transformers import AutoModelForCausalLM
import sys
import os  
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
import os
import torch


def eval_ni(
    config,
    model,
    nshot=2,
    data_dir=None,
    subsample=-1,
):
    from mttl.evaluators import NIEvaluator

    ni_evaluator = NIEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
        num_pos_examples=nshot
    )
    metrics = ni_evaluator.evaluate(model,subsample)
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    from huggingface_hub import login

    config = RoutingConfig.parse(c="/home/v-oostapenko/dev/mttl/projects/instr_routing/configs/alpaca/llama1_7b_vsmear.json")

    login(token=os.environ["HUGGING_FACE_HUB_TOKEN"])
    config.data_dir = os.environ["NI_DATA_DIR"]
    dm = AlpacaDataModule(config)
    path_best_model = "/home/v-oostapenko/dev/mttl/tmp/instruction_learning/yahma_llama-7b-hf0qx192oq_None-val/loss=1.4099.ckpt"
    best_model = CLM.load_from_checkpoint(path_best_model, tokenizer=dm.tokenizer).cuda()
      
    print(eval_ni(config, best_model, nshot=0, subsample=10))
