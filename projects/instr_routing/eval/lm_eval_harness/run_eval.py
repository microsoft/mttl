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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
   
import glob       
# from mttl.models.poly import get_selector
from mttl.models.utils import RoutingInfo  
from inst_follow.eval.gen_ni_predictions import load_model_for_generation, dict_to_dataclass

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
def main(save_dir="/home/v-oostapenko/results/mmlu/llama-7B/", 
         model_name="",
         model_path="",     
         amlt_experiment_name = "alpaca_smear", task="hellaswag", nshot = 0, batch_size=5):                   
    return eval_lm(save_dir, model_name, model_path, amlt_experiment_name, task, nshot, batch_size)

def eval_lm(save_dir="/home/v-oostapenko/results/mmlu/llama-7B/", 
        model_name="",   
        model_path="", amlt_experiment_name="",
        task="hellaswag", nshot = 0, batch_size=5):            
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:   
        save_dir = os.environ.get("AMLT_OUTPUT_DIR")
        data_dir="/mnt/default/data/mmlu/" # on gcr        
        model_dict = {           
            "alpaca_poly_1": {"from_hf":0, "model_name":"alpaca_poly_1", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hf7btqc8tq_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4240.ckpt/loss=0.4240.ckpt"},
            "alpaca_poly_2": {"from_hf":0, "model_name":"alpaca_poly_2", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfz3sxro0n_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4244.ckpt/loss=0.4244.ckpt"},
            "alpaca_poly_3": {"from_hf":0, "model_name":"alpaca_poly_3", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfz5pqv3xm_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
            "alpaca_poly_4": {"from_hf":0, "model_name":"alpaca_poly_4", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hftoqv8su7_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
            "alpaca_poly_5": {"from_hf":0, "model_name":"alpaca_poly_5", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hflpwsu1yg_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4248.ckpt/loss=0.4248.ckpt"},     
            "alpaca_poly_6": {"from_hf":0, "model_name":"alpaca_poly_6", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hf5y0pj3u9_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4215.ckpt/loss=0.4215.ckpt"},   
            # "alpaca_cbtm_dist4_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist4_ms0_temp01", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfsx5tnvyw_alpaca_lora_cbtm_dist-val/best_model_alpaca_lora_cbtm_dist_loss=0.4238.ckpt/loss=0.4238.ckpt"},
            # "alpaca_cbtm_dist4_ms0_temp1": {"from_hf":0, "model_name":"alpaca_cbtm_dist4_ms0_temp1", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfdu6k7bfe_alpaca_lora_cbtm_dist-val/best_model_alpaca_lora_cbtm_dist_loss=0.4206.ckpt/loss=0.4206.ckpt"},
            # "alpaca_cbtm_dist8_ms0_temp1": {"from_hf":0, "model_name":"alpaca_cbtm_dist8_ms0_temp1", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfrzg373sg_alpaca_lora_cbtm_dist-val/best_model_alpaca_lora_cbtm_dist_loss=0.4205.ckpt/loss=0.4205.ckpt"},
            # "alpaca_cbtm_dist8_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist8_ms0_temp01", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hf70gdhzyd_alpaca_lora_cbtm_dist-val/best_model_alpaca_lora_cbtm_dist_loss=0.4282.ckpt/loss=0.4282.ckpt"},        
            # "alpaca_cbtm_dist16_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist16_ms0_temp01", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfuu7ix6gf_alpaca_lora_cbtm_dist-val/best_model_alpaca_lora_cbtm_dist_loss=0.4303.ckpt/loss=0.4303.ckpt"},
            # "alpaca_cbtm_dist32_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist32_ms0_temp01", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hf435zt6xt_alpaca_lora_cbtm_dist-val/loss=0.4365.ckpt"},
            "alpaca_cbtm_dense": {"from_hf":0, "model_name":"alpaca_cbtm_dense", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfd5p811ne_alpaca_lora_cbtm_dense-val/best_model_alpaca_lora_cbtm_dense_loss=0.4215.ckpt/loss=0.4215.ckpt"},
            "llama_flanv2_cbtm_dense": {"from_hf":0, "model_name":"llama_flanv2_cbtm_dense", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dense/yahma_llama-7b-hfi5u2647s_alpaca_lora_cbtm_dense-val/loss=0.0782.ckpt"},       
            "llama_flanv2_cbtm_dense_lr_low": {"from_hf":0, "model_name":"llama_flanv2_cbtm_dense_lr_low", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dense_lr_low/yahma_llama-7b-hfjsdsv59e_alpaca_lora_cbtm_dense-val/loss=0.0845.ckpt"},
            "llama_flanv2_cbtm_dense_R8": {"from_hf":0, "model_name":"llama_flanv2_cbtm_dense_R8", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dense_R8/yahma_llama-7b-hfsid8qkv2_alpaca_lora_cbtm_dense-val/loss=0.0781.ckpt"}, 
            "llama_flanv2_cbtm_dist4_ms1_temp01": {"from_hf":0, "model_name":"llama_flanv2_cbtm_dist4_ms1_temp01", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist4_ms1_temp01/yahma_llama-7b-hf0c5zdnfj_alpaca_lora_cbtm_dist-val/loss=0.0691.ckpt"},
            "llama_flanv2_cbtm_dense_R100": {"from_hf":0, "model_name":"llama_flanv2_cbtm_dense_R100", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dense_R100/yahma_llama-7b-hfe0kj95a9_alpaca_lora_cbtm_dense-val/loss=0.0780.ckpt"}, 
            "llama_flanv2_cbtm_dist8_ms1_temp01":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist8_ms1_temp01", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist8_ms1_temp01/yahma_llama-7b-hfv2y7w3oc_alpaca_lora_cbtm_dist-val/loss=0.0690.ckpt"},
            "llama_flanv2_cbtm_dist16_ms1_temp01":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist16_ms1_temp01", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist16_ms1_temp01/yahma_llama-7b-hfrd8rdv4k_alpaca_lora_cbtm_dist-val/loss=0.0693.ckpt"},   
            "llama_flanv2_cbtm_dist4_ms1_temp01_R8": {"from_hf":0, "model_name":"llama_flanv2_cbtm_dist4_ms1_temp01_R8", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist4_ms1_temp01_R8/yahma_llama-7b-hffdouag0v_alpaca_lora_cbtm_dist-val/loss=0.0689.ckpt"},
            "llama_flanv2_cbtm_dist8_ms1_temp01_R8":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist8_ms1_temp01_R8", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist8_ms1_temp01_R8/yahma_llama-7b-hfiyprlomq_alpaca_lora_cbtm_dist-val/loss=0.0690.ckpt"},
            "llama_flanv2_cbtm_dist16_ms1_temp01_R8":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist16_ms1_temp01_R8", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist16_ms1_temp01_R8/yahma_llama-7b-hftz6msdn6_alpaca_lora_cbtm_dist-val/loss=0.0691.ckpt"},
            "llama_flanv2_cbtm_dist4_ms1_temp01_R8_shuffled":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist4_ms1_temp01_R8_shuffled", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist4_ms1_temp01_R8_shuffled/yahma_llama-7b-hf894p3mob_alpaca_lora_cbtm_dist-val/loss=0.0705.ckpt"},
            "llama_flanv2_cbtm_dist8_ms1_temp01_R8_shuffled":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist8_ms1_temp01_R8_shuffled", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist8_ms1_temp01_R8_shuffled/yahma_llama-7b-hf0jgv7hr2_alpaca_lora_cbtm_dist-val/loss=0.0720.ckpt"},
            "llama_flanv2_cbtm_dense_mil512":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dense_mil512", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dense_mil512/yahma_llama-7b-hfyhhei3lb_alpaca_lora_cbtm_dense-val/loss=0.0689.ckpt"},
            "llama_flanv2_cbtm_dense_R100_mil512":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dense_R100_mil512", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dense_R100_mil512/yahma_llama-7b-hf1lgcsbq6_alpaca_lora_cbtm_dense-val/loss=0.0687.ckpt"},
            "llama_flanv2_cbtm_dist4_ms0_temp01_R8":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist4_ms0_temp01_R8", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist4_ms0_temp01_R8/yahma_llama-7b-hfvdddphil_alpaca_lora_cbtm_dist-val/loss=0.0688.ckpt"},
            "llama_flanv2_cbtm_dist8_ms0_temp01_R8":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist8_ms0_temp01_R8", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist8_ms0_temp01_R8/yahma_llama-7b-hfagyanptl_alpaca_lora_cbtm_dist-val/loss=0.0687.ckpt"},
            "llama_flanv2_cbtm_dist16_ms0_temp01_R8":{"from_hf":0, "model_name":"llama_flanv2_cbtm_dist16_ms0_temp01_R8", "depth":0, "model_path":"/mnt/default/data/models/alpaca_flan_v2_subset/llama_flanv2_cbtm_dist16_ms0_temp01_R8/yahma_llama-7b-hf54x0h1ul_alpaca_lora_cbtm_dist-val/loss=0.0695.ckpt"},
            # lora_vs_full    
            # "lora_vs_full_allpaca_full_lower_lr": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_rank_allpaca_full_lower_lr/yahma*/loss=*.ckpt")[0]},    
            # "lora_vs_full_lora_r100": {"from_hf":0, "model_name":"lora_vs_full_lora_r100", "depth":0, "model_path": "/mnt/default/data/models/alpaca_rank_allpaca_lora_r100/yahma_llama-7b-hf4xne2yrp_alpaca_full-val/loss=0.4280.ckpt"},
            # smear 
            "smear_dense_alpaca": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_lora_cbtm_clustering/alpaca_cbtm_dense/yahma*/loss=*.ckpt")[0]},
            "smear_4s_alpaca_pt": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_smear/alpaca_smear_4_pt/yahma*/loss=*.ckpt")[0]},
            "smear_8s_alpaca_pt": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_smear/alpaca_smear_8_pt/yahma*/loss=*.ckpt")[0]},
            "smear_4s_alpaca_pe": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_smear/alpaca_smear_4_pe/yahma*/loss=*.ckpt")[0]},
            "smear_8s_alpaca_pe": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_smear/alpaca_smear_8_pe/yahma*/loss=*.ckpt")[0]},        
            # alpaca tfidf
            "alpaca_cbtm_dist4_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist4_ms0_temp01", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist4_ms0_temp01/yahma_llama-7b-hfa5oww435_alpaca_lora_cbtm_dist-val/loss=*.ckpt")[0]},
            "alpaca_cbtm_dist8_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist8_ms0_temp01", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist8_ms0_temp01/yahma_llama-7b-hfcyd9xh2d_alpaca_lora_cbtm_dist-val/loss=*.ckpt")[0]},
            "alpaca_cbtm_dist8_ms0_temp01_w_outputs": {"from_hf":0, "model_name":"alpaca_cbtm_dist8_ms0_temp01_w_outputs", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist8_ms0_temp01_w_outputs/yahma*/loss=*.ckpt")[0]},
            "alpaca_cbtm_dist8_ms0_temp01_permuted": {"from_hf":0, "model_name":"alpaca_cbtm_dist8_ms0_temp01_permuted", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist8_ms0_temp01_permuted/yahma*/loss=*.ckpt")[0]},
            "alpaca_cbtm_dist4_ms0_temp01_permuted": {"from_hf":0, "model_name":"alpaca_cbtm_dist4_ms0_temp01_permuted", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist4_ms0_temp01_permuted/yahma*/loss=*.ckpt")[0]},
            "alpaca_cbtm_dist4_ms0_temp01_w_outputs": {"from_hf":0, "model_name":"alpaca_cbtm_dist4_ms0_temp01_w_outputs", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist4_ms0_temp01_w_outputs/yahma*/loss=*.ckpt")[0]},
            "alpaca_smear_8_pe_long_kminit": {"from_hf":0, "model_name":"alpaca_smear_8_pe_long_kminit", "depth":0, "model_path":glob.glob("/mnt/default/data/models/alpaca_smear/alpaca_smear_8_pe_long_kminit/yahma_llama-7b-hfx39ys94j_alpaca_lora_cbtm_dense-val/loss=*.ckpt")[0]},
            "alpaca_finetune_router_from_LDA8": {"from_hf":0, "model_name":"alpaca_finetune_router_from_LDA8", "depth":0, "model_path":"/mnt/default/data/models/tmp/instruction_learning/yahma_llama-7b-hfjtjurxmy_alpaca_em_smear_8_pe_from_LDA_initial-val/loss=0.8511.ckpt"},

        }  
        base_model_path = "/mnt/default/data/models"    
    # data_path = os.getenv("AP_DATA_DIR", "/home/v-oostapenko/dev/natural-instructions/tasks")
    else:
        data_dir = "/home/v-oostapenko/data/mmlu"
        base_model_path = "/home/v-oostapenko/dev/amlt/"                
        model_dict = { 
            "alpaca_poly_1": {"from_hf":0, "model_name":"alpaca_poly_1", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly1/yahma_llama-7b-hf7btqc8tq_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4240.ckpt/loss=0.4240.ckpt"},
            "alpaca_poly_2": {"from_hf":0, "model_name":"alpaca_poly_2", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly2/yahma_llama-7b-hfz3sxro0n_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4244.ckpt/loss=0.4244.ckpt"},
            "alpaca_poly_3": {"from_hf":0, "model_name":"alpaca_poly_3", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly3/yahma_llama-7b-hfz5pqv3xm_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
            "alpaca_poly_4": {"from_hf":0, "model_name":"alpaca_poly_4", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly4/yahma_llama-7b-hftoqv8su7_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
            "alpaca_poly_5": {"from_hf":0, "model_name":"alpaca_poly_5", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly5/yahma_llama-7b-hflpwsu1yg_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4248.ckpt/loss=0.4248.ckpt"},     
            "alpaca_poly_6": {"from_hf":0, "model_name":"alpaca_poly_6", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly6/yahma_llama-7b-hf5y0pj3u9_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4215.ckpt/loss=0.4215.ckpt"},  
            # "alpaca_cbtm_dist4_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist4_ms0_temp01", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist4_ms0_temp01/yahma_llama-7b-hfsx5tnvyw_alpaca_lora_cbtm_dist-val/best_model_alpaca_lora_cbtm_dist_loss=0.4238.ckpt/loss=0.4238.ckpt"},
            "alpaca_cbtm_dense": {"from_hf":0, "model_name":"alpaca_cbtm_dense", "depth":0, "model_path":"/home/v-oostapenko/dev/mttl/mmodels_gcr/yahma_llama-7b-hfd5p811ne_alpaca_lora_cbtm_dense-val/best_model_alpaca_lora_cbtm_dense_loss=0.4215.ckpt/loss=0.4215.ckpt"},
            # ~/dev/amlt/lora_vs_full
            "lora_vs_full_allpaca_full": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full", "depth":0, "model_path":glob.glob("/home/v-oostapenko/dev/amlt/lora_vs_full/alpaca_rank_allpaca_full/yahma*/loss=*.ckpt")[0]},
            "lora_vs_full_lora_r100": {"from_hf":0, "model_name":"lora_vs_full_lora_r100", "depth":0, "model_path":glob.glob("/home/v-oostapenko/dev/amlt/lora_vs_full/alpaca_rank_allpaca_lora_r100/yahma*/loss=*.ckpt")[0]},   
            "alpaca_cbtm_dist4_ms0_temp01": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/home/v-oostapenko/dev/amlt/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist4_ms0_temp01/yahma_llama-7b-hfa5oww435_alpaca_lora_cbtm_dist-val/loss=*.ckpt")[0]},
            "smear_8s_alpaca_pe": {"from_hf":0, "model_name":"lora_vs_full_allpaca_full_lower_lr", "depth":0, "model_path":glob.glob("/home/v-oostapenko/dev/amlt/alpaca_smear/alpaca_smear_8_pe/yahma*/loss=*.ckpt")[0]},
            "alpaca_smear_8_pe_long_kminit": {"from_hf":0, "model_name":"smear_8_pr_long_kminit", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_smear/alpaca_smear_8_pe_long_kminit/yahma_llama-7b-hfx39ys94j_alpaca_lora_cbtm_dense-val/loss=0.8162.ckpt"},        
            "alpaca_finetune_router_from_LDA8": {"from_hf":0, "model_name":"alpaca_finetune_router_from_LDA8", "depth":0, "model_path":"/home/v-oostapenko/dev/mttl/inst_follow/tmp/instruction_learning/yahma_llama-7b-hfj8fdwq5x_alpaca_em_smear_8_pe_from_LDA_initial-val/loss=0.8909.ckpt"},
            "alpaca_smear_4_pe_kaiming_padtokmask_learn_router": {"from_hf":0, "model_name":"alpaca_smear_4_pe_kaiming_padtokmask_learn_router", "depth":0, "model_path":glob.glob("/home/v-oostapenko/dev/amlt/alpaca_smear/alpaca_smear_4_pe_kaiming_padtokmask_learn_router/yahma*/loss=*.ckpt")[0]},

        }
    
    if model_path is None:
        if model_name in model_dict:
            model_path = model_dict[model_name]["model_path"]
            model_name = model_dict[model_name]["model_name"]
        else:      
            exp_name = amlt_experiment_name
            from_hf = 0
            model_path = glob.glob(f"{base_model_path}/{exp_name}/{model_name}/yahma*/loss=*.ckpt")[0]
    else:
       from_hf = 0 # model path is given, not using hf
       model_name = ""
    
    
    base_model = "yahma/llama-7b-hf"             
    model, tokenizer, config, topic_router = load_model_for_generation(from_hf, base_model, 
                                                                       model_name, 
                                                                       model_path, "")
    
    from lm_eval import tasks, evaluator, utils
    from lm_eval.models import MODEL_REGISTRY
    HFLM = MODEL_REGISTRY["hf-causal"]
            
    class HFLM_Custom(HFLM):    
        def _model_call(self, inps):                                
            self.model.task_id_container["routing_infos"] = RoutingInfo(task_ids=-1*torch.ones(inps.shape[0]).to(self.model.device),
                                                                        hashes=None, 
                                                                        pad_token_mask = (inps!=0).float().to(self.model.device))
            
            return super()._model_call(inps)
        
        def _model_generate(self, context, max_length, eos_token_id):   
            self.model.task_id_container["routing_infos"] = RoutingInfo(task_ids=-1*torch.ones(context.shape[0]).to(self.model.device),
                                                                        hashes=None, 
                                                                        pad_token_mask = (context!=0).float().to(self.model.device))
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
        limit=None, #0.01,  
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