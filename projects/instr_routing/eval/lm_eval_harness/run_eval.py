import argparse
import os 
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
import sys 
import click   

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from projects.instr_routing.eval.model_dict import model_dict   
import glob       
# from mttl.models.poly import get_selector
from projects.instr_routing.models.utils import RoutingInfo 
from projects.instr_routing.eval.ni.gen_ni_predictions import load_model_for_generation, dict_to_dataclass

@click.command()    
# @click.option("--ntrain", type=int, default=5)        
@click.option("--save_dir", type=str, default="results/mmlu/llama-7B/")
@click.option(
    "--model_name",
    type=str,
    default="platypus_smear_4_xr4",
    help="if specified, we will load the model to generate the predictions.",
)          
@click.option( 
    "--model_path",
    type=str,            
    default='/home/v-oostapenko/dev/amlt/alpaca_smear/platypus_smear_8_xr4_cos/yahma_llama-7b-hf4wtmnua3_platypus_dense-val/loss=0.7795.ckpt',
    help="if specified, we will load the model to generate the predictions.",
)   
@click.option("--amlt_experiment_name", type=str, default="alpaca_smear")
@click.option("--task", type=str, default="truthfulqa_mc")
@click.option("--nshot", type=int, default=0)       
@click.option("--batch_size", type=int, default=5)
@click.option("--ds_limit", type=float, default=None)  
def main(save_dir="/home/v-oostapenko/results/mmlu/llama-7B/", 
         model_name="",
         model_path="",      
         amlt_experiment_name = "alpaca_smear", task="hellaswag", nshot = 0, batch_size=5, ds_limit=None):                   
    return eval_lm(save_dir, model_name, model_path, amlt_experiment_name, task, nshot, batch_size, ds_limit)

def eval_lm(save_dir="/home/v-oostapenko/results/mmlu/llama-7B/", 
        model_name="",   
        model_path="",     
        amlt_experiment_name="", 
        task="hellaswag", 
        nshot = 0, 
        batch_size=5,
        ds_limit=None):  
    
    from_hf = 0  # TODO: make sure this can also eval model from hf
    ################################################################################################
    # set paths
    code_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    save_dir = os.getenv("AMLT_OUTPUT_DIR", save_dir)
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:  # on gcr
        data_dir = "/mnt/default/data/mmlu/"
        base_model_path = "/mnt/default/data/models"
        base_cluster_infos = "/mnt/default/data/"  # /mnt/amlt_code/inst_follow/"
    else:
        base_cluster_infos = code_dir
    path_to_clusterer = f"{base_cluster_infos}/cluster_infos/cbtm/"
    ################################################################################################

    if model_path == "" and from_hf == 0:
        if model_name in model_dict:
            # can also list models in the model dict
            from_hf = model_dict[model_name]["from_hf"]
            model_path = model_dict[model_name]["model_path"]
            out_prefix = model_name
        elif not from_hf:
            exp_name = amlt_experiment_name
            from_hf = 0
            model_path = glob.glob(
                f"{base_model_path}/{exp_name}/{model_name}/yahma*/loss=*.ckpt"
            )[0]
            out_prefix = model_name
            
    base_model = "yahma/llama-7b-hf" 
    model, tokenizer, config, topic_router = load_model_for_generation(
        from_hf, base_model, model_name, model_path, None, code_dir=code_dir
    )
    
    
    
    from lm_eval import tasks, evaluator, utils
    from lm_eval.models import MODEL_REGISTRY
    HFLM = MODEL_REGISTRY["hf-causal"]
            
    class HFLM_Custom(HFLM):     
        def _model_call(self, inps):                                
            self.model.task_id_container["routing_infos"] = RoutingInfo(task_ids=-1*torch.ones(inps.shape[0]).to(self.model.device),
                                                                        hashes=None, 
                                                                        pad_token_mask = (inps!=0).float().to(self.model.device))
            setattr(self.model.task_id_container["routing_infos"], "gen_mode", True)
            
            return super()._model_call(inps)
        
        def _model_generate(self, context, max_length, eos_token_id):      
            self.model.task_id_container["routing_infos"] = RoutingInfo(task_ids=-1*torch.ones(context.shape[0]).to(self.model.device),
                                                                        hashes=None, 
                                                                        pad_token_mask = (context!=0).float().to(self.model.device))
            setattr(self.model.task_id_container["routing_infos"], "gen_mode", True)
            return super()._model_generate(context, max_length, eos_token_id)
      
    lm_eval_model = HFLM_Custom(pretrained=model.model,tokenizer=tokenizer)
    
    results = evaluator.simple_evaluate(
        model=lm_eval_model, 
        model_args="",           
        tasks=[task],  
        num_fewshot=nshot,  
        batch_size=batch_size,   
        max_batch_size=None,
        device="cuda",
        no_cache=False,
        limit=ds_limit,  
        description_dict=None,     
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        output_base_path=save_dir
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)      
    save_dir = save_dir + f"/{task}_{model_name}_{nshot}.json"
    if save_dir:  
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        with open(save_dir, "w") as f:
            f.write(dumped)

    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    # print(
    #     f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
    #     f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    # )
    print(evaluator.make_table(results))
    results_dict = {} 
    r = results['results'][task]
    for k, v in r.items():
        k+=f"_{task}"    
        results_dict[k] = v
    del lm_eval_model, model
    # clean cache
    torch.cuda.empty_cache()
    return results_dict
    

        
if __name__ == "__main__":
    main()