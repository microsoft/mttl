import json
import os 
import sys
import glob
import numpy as np
import click
import json
import tqdm
import copy
import torch
import pickle
import datasets
from types import SimpleNamespace
from sklearn.feature_extraction.text import TfidfVectorizer
# sys.path.append("/home/v-oostapenko/dev/mttl")
directory = os.getenv("CODE_DIR", "/home/v-oostapenko/dev/mttl")
sys.path.append(directory)
sys.path.append("../..") 
sys.path.append("/mnt/amlt_code/")
from inst_follow.models.clm import CLM  
from transformers import LlamaTokenizer  
from inst_follow.eval.gen_ni_predictions import load_model_for_generation
from mttl.models.poly import get_selector     
from mttl.models.modify_model import modify_transformer  
from finetune_llama import parse_config, Config                  
from inst_follow.utils import load_model, TopicRouter,disable_torch_init
from mttl.cluster_tuning.cluster_reader import ClusterResult
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
def dict_to_dataclass(d):
    from dataclasses import make_dataclass
    return make_dataclass("X", d.keys())(**d)


def prompt(instruction):
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction:\
            {instruction}\
            \n### Response:"

def get_outputs(input, model, tokenizer, do_sample, temperature, max_output_length, top_p):
        otuputs_list=[]      
        # eval with the best adapter for this cluster
        output_ids = model.generate(
            input,
            do_sample=do_sample,
            temperature=temperature,      
            max_new_tokens=max_output_length,
            top_p=top_p,
            # top_k=50,
            return_dict_in_generate=True,
        )
        # for out,ex in zip(output_ids.sequences, examples):
        #     # out = out[len(ex):] 
        #     output_str = tokenizer.decode(out)
        #     otuputs_list.append(output_str)
        outputs = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)   
        examples_in_decoded = tokenizer.batch_decode(input["input_ids"], skip_special_tokens=True) # just in case, to make sure inputs are exactly the same as expected here when spliting the decodings
        for out,ex in zip(outputs, examples_in_decoded):
            o = out.split(ex)
            assert len(o)>1 
            otuputs_list.append(o[1])
        # output_cleaned = [out.split(ex)[1] for out,ex in zip(outputs, examples)]           
        # otuputs_list.extend(output_cleaned)
        del output_ids
        del input  
        return otuputs_list


def example_in_cluster(text, vectorizer, kmeans, random_clusters=False, distances = False):
    if distances:
        from kmeans_pytorch import pairwise_distance
        centers = kmeans.cluster_centers
        distances_to_centers = pairwise_distance(torch.from_numpy(vectorizer.transform(text)).cuda(), centers)
        return list(distances_to_centers)
    if random_clusters:     
        clusters = np.random.choice(range(kmeans.n_clusters), len(text))
    else:
        clusters = kmeans.predict(torch.from_numpy(vectorizer.transform(text)))
    return list(clusters)

def number_normalizer(tokens):
    """Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)

   
class NumberNormalizingVectorizer(TfidfVectorizer):
    # this vectorizer replaces numbers with #NUMBER token
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()   
        return lambda doc: list(number_normalizer(tokenize(doc)))    

def load_kmeans_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out

@torch.no_grad() 
def generate_outputs(model,           
                     examples, tokenizer,   
                     temperature=0.7, do_sample=False, 
                     max_output_length=512, top_p=1.0, 
                     topic_router=None, 
                     skill_selector="topic",
                     lda_depth=1, prompt_before_lda=True, run_all_clusters=False, kmeans=None, tfidf=None):
    # otuputs_list=[]               
    if isinstance(examples, str):
        examples = [examples]
    if prompt_before_lda:
        for i,ex in enumerate(examples):
            examples[i] = prompt(ex)
    probs=None
    if topic_router:      
        if skill_selector=="random":
            # random binary matrix 
            raise NotImplementedError()
            input["distances"] = torch.randint(0,2,(len(examples), topic_router.n_skills)).cuda()
        else: 
            if not run_all_clusters:
                probs = topic_router(examples, depth=lda_depth)   
                if hasattr(model, "skill_ids_to_keep"):
                    probs = probs[:,model.skill_ids_to_keep]
                if hasattr(model, "shared_adapter"):
                    probs[:,model.shared_adapter] = probs.sum(-1)/2
                    # renormalize
                    probs=probs[:,:]/probs.sum(-1)[...,None]
            else:
                # just all eros 
                probs = torch.ones(len(examples), topic_router.n_skills).cuda()
            # input["distances"] = probs
    
    elif kmeans is not None and tfidf is not None:   
        flag=0
        if len(examples)==1:
            flag=1   
            examples=[examples[0],examples[0]]
        cluster = example_in_cluster(examples,  tfidf, kmeans, random_clusters=False, distances=True)
        if flag:   
            cluster=[cluster[0]]     
        probs = torch.stack(cluster).cuda()   
        
    if not prompt_before_lda:
        for i,ex in enumerate(examples):
            examples[i] = prompt(ex)
            
    inputs = tokenizer(examples,
            padding='longest',
            return_tensors="pt")    
    input={                         
        "input_ids": inputs.input_ids.to(device),    
        "task_ids": torch.zeros(len(examples), dtype=torch.long).to(device)*-1,
    }  
    input["pad_token_mask"] = (inputs.input_ids != tokenizer.pad_token_id).float().to(device)
    
    if probs is not None:
        if not run_all_clusters:
            input["distances"] = probs 
            return get_outputs(input, model, tokenizer, do_sample, temperature, max_output_length, top_p)
        else:
            otuputs_dict = {}
            for c in range(probs.shape[1]): 
                input["distances"] = torch.zeros(probs.shape)
                input["distances"][:,c] = 1
                out_c = get_outputs(input, model, tokenizer, do_sample, temperature, max_output_length, top_p)
                otuputs_dict[c] = out_c                
            return otuputs_dict
          
    
    return get_outputs(input, model, tokenizer, do_sample, temperature, max_output_length, top_p)

          
@click.command()                  
@click.option("--llama_model", type=str, default="yahma/llama-7b-hf") #chavinlo/alpaca-native") yahma/llama-7b-hf chainyo/alpaca-lora-7b togethercomputer/RedPajama-INCITE-Base-7B-v0.1
@click.option("--batch_size", type=int, default=1)         
@click.option("--model_name", type=str, default="alpaca_smear_dense100r")     
@click.option("--from_hf", type=int, default=0)
@click.option("--model_path", type=str, default="") #/home/v-oostapenko/logs/model_from_1016/amlt_yahma_llama_atlas_cluster_l1/alpaca4r_topic_ldal1/alpaca-lora_l1/best_model_alpaca_lora_atlas_cluster_te_ada_l1/loss=0.4242.ckpt")#"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hfopq9a3dw_alpaca_lora_full-val/loss=0.5940.ckpt")
# l1 lda: /home/v-oostapenko/logs/model_from_1016/amlt_yahma_llama_atlas_cluster_l1/alpaca4r_topic_ldal1/alpaca-lora_l1/best_model_alpaca_lora_atlas_cluster_te_ada_l1/loss=0.4242.ckpt
# @click.option("--example_to_ids_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002.pkl") # depreciated
@click.option("--skill_selector", type=str, default="topic")
@click.option("--prompt_before_lda", type=bool, default=False)
@click.option("--run_all_clusters", type=bool, default=False)
@click.option("--of_pref", type=str, default="")       
@click.option("--cbtm_n_clusters", type=int, default=-1)
@click.option("--amlt_experiment_name", type=str, default="alpaca_smear")   
def gen_alpaca_eval_command(llama_model="gpt3", batch_size=4, model_name="", from_hf=0, 
                            model_path="", skill_selector="topic", prompt_before_lda=True, of_pref="", run_all_clusters=True, cbtm_n_clusters=-1, amlt_experiment_name="alpaca_smear"):
    return gen_alpaca_evl(llama_model, batch_size, model_name, from_hf, model_path, skill_selector, prompt_before_lda, of_pref, run_all_clusters, cbtm_n_clusters, amlt_experiment_name)

def gen_alpaca_evl(llama_model="gpt3",      
         batch_size=4, model_name="", 
         from_hf=0, model_path="", 
         skill_selector="topic", 
         prompt_before_lda=True, 
         of_pref="",                          
         run_all_clusters=True, cbtm_n_clusters=-1, amlt_experiment_name="alpaca_smear"):
    
    # home=os.getenv("CONFIG_DIR","/home/v-oostapenko/dev/")
            
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:   
        data_path="/mnt/default/data/natural-instructions/tasks" # on gcr
        
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
        data_path = "/home/v-oostapenko/dev/natural-instructions/tasks"
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
                   
        
    shared_adapter=False
    do_sample=False
    top_p=1.0
    max_output_length=2048
    temperature=0.7
    depth=1
    if model_name in model_dict:
        from_hf = model_dict[model_name]["from_hf"]
        model_path = model_dict[model_name]["model_path"]
        depth = model_dict[model_name]["depth"]
        model_name = model_dict[model_name]["model_name"]
        if "do_sample" in model_dict[model_name]:
            do_sample = model_dict[model_name]["do_sample"]
        if "top_p" in model_dict[model_name]:
            top_p = model_dict[model_name]["top_p"]
        if "max_new_tokens" in model_dict[model_name]:
            max_output_length = model_dict[model_name]["max_new_tokens"]
        # if "shared_adapter" in model_dict[model_name]:
        #     shared_adapter = model_dict[model_name]["shared_adapter"]
    else:      
        exp_name = amlt_experiment_name
        from_hf = 0
        model_path = glob.glob(f"{base_model_path}/{exp_name}/{model_name}/yahma*/loss=*.ckpt")[0]
        # out_prefix = model_name
    
        
    task_results = {} 
    topic_router = None
    disable_torch_init()      
    
    base_model = llama_model             
    model, tokenizer, config, topic_router = load_model_for_generation(from_hf, base_model, model_name, model_path, skill_selector)
    
    
    base = os.getenv("AMLT_OUTPUT_DIR", "/home/v-oostapenko/dev/mttl/inst_follow")
    base_cluster_infos="/home/v-oostapenko/dev/mttl/inst_follow/"
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:
        base_cluster_infos  = "/mnt/amlt_code/inst_follow/"
    path_to_clusterer = f"{base_cluster_infos}/cluster_infos/cbtm/"
    
    ############
    # configure router
    tfidf, kmeans = None, None
    # infer what clusterer we are using
    if config.example_to_ids_path is not None:
        if "kmeans" in config.example_to_ids_path:
            cbtm_n_clusters = ClusterResult(config.example_to_ids_path).n_clusters()
            dataset = config.dataset
    
    if cbtm_n_clusters>0:
        tdifd = "tfidf"
        clusterer=f"kmeans{cbtm_n_clusters}"      
        if dataset=="flan_v2":
            clusterer = f"kmeans_flan_v2{cbtm_n_clusters}"
            tdifd = "tfidf_flv2"
        tfidf = load_kmeans_model(path_to_clusterer+f"/{tdifd}.pkl")
        kmeans = load_kmeans_model(path_to_clusterer+f"/{clusterer}.pkl")
        topic_router = None
    
    
    # print(config) 
    print("Arguments:\n")         
    print(llama_model, "\n", batch_size, "\n", model_name, "\n", from_hf, "\n", model_path, "\n", skill_selector)
    # nshot = 1    
    out_file_name = f"{of_pref}al_eval_pred_[{model_name}].json"
    out_file_name=out_file_name.replace("/", "_")
    print("Output file name:", out_file_name)
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    examples = [] 
    bs = batch_size
    batch = []
    examples_batch=[]
    outptut_examples = []
    for example in eval_set:   
        if len(batch)<bs:    
            batch.append(example["instruction"])
            examples_batch.append(example)
        else:
            batch.append(example["instruction"])    
            examples_batch.append(example)              
            outs = generate_outputs(model,batch, tokenizer, 
                                    temperature=temperature, 
                                    do_sample=do_sample, 
                                    max_output_length=max_output_length,  
                                    top_p=top_p, topic_router=topic_router, 
                                    skill_selector=skill_selector, 
                                    lda_depth=depth, prompt_before_lda=prompt_before_lda, run_all_clusters=run_all_clusters, kmeans=kmeans, tfidf=tfidf)
            if isinstance(outs, dict):
                keys = list(outs.keys())
                _examples_batch= []
                for k in keys:            
                    for ex,out in zip(examples_batch,outs[k]):
                        ex["output"] = out 
                        ex["generator"]=model_name+"_c"+str(k)
                        _examples_batch.append(copy.deepcopy(ex))
                examples_batch = _examples_batch
            else:    
                for ex,out in zip(examples_batch,outs):
                    ex["output"] = out
                    ex["generator"]=model_name  
            outptut_examples.extend(examples_batch)     
            # create riectories if dont exist
            if not os.path.exists(f"{base}/eval/alpaca_eval/"):  # Check if the path already exists
                os.makedirs(f"{base}/eval/alpaca_eval/") 
                     
            with open(f"{base}/eval/alpaca_eval/{out_file_name}", 'w') as file:
                json.dump(outptut_examples, file, indent=4)          
            
            
            batch=[]    
            examples_batch=[]
    if len(batch)>0:
        outs = generate_outputs(model,batch, tokenizer, 
                                temperature=temperature, 
                                do_sample=do_sample, 
                                max_output_length=max_output_length, 
                                top_p=top_p, topic_router=topic_router, 
                                skill_selector=skill_selector, 
                                lda_depth=depth, prompt_before_lda=prompt_before_lda, run_all_clusters=run_all_clusters, kmeans=kmeans, tfidf=tfidf)        
        if isinstance(outs, dict):
            keys = list(outs.keys())
            _examples_batch= []
            for k in keys:
                for ex,out in zip(examples_batch,outs[k]):
                    ex["output"] = out 
                    ex["generator"]=model_name+"_c"+str(k)
                    _examples_batch.append(copy.deepcopy(ex))
            examples_batch = _examples_batch
        else:    
            for ex,out in zip(examples_batch,outs):
                ex["output"] = out
                ex["generator"]=model_name
        outptut_examples.extend(examples_batch)    
        with open(f"{base}/eval/alpaca_eval/{out_file_name}", 'w') as file:
            json.dump(outptut_examples, file, indent=4)
    
            
if __name__ == '__main__':
    gen_alpaca_eval_command()