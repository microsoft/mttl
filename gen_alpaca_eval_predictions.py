import json
import os 
import sys
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
@click.option("--model_name", type=str, default="alpaca_smear_8_pe_long_kminit")     
@click.option("--from_hf", type=int, default=3)
@click.option("--model_path", type=str, default="/home/v-oostapenko/logs/model_from_1016/amlt_yahma_llama_atlas_cluster_l1/alpaca4r_topic_ldal1/alpaca-lora_l1/best_model_alpaca_lora_atlas_cluster_te_ada_l1/loss=0.4242.ckpt")#"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hfopq9a3dw_alpaca_lora_full-val/loss=0.5940.ckpt")
# l1 lda: /home/v-oostapenko/logs/model_from_1016/amlt_yahma_llama_atlas_cluster_l1/alpaca4r_topic_ldal1/alpaca-lora_l1/best_model_alpaca_lora_atlas_cluster_te_ada_l1/loss=0.4242.ckpt
# @click.option("--example_to_ids_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002.pkl") # depreciated
@click.option("--skill_selector", type=str, default="topic")
@click.option("--prompt_before_lda", type=bool, default=False)
@click.option("--run_all_clusters", type=bool, default=False)
@click.option("--of_pref", type=str, default="")       
@click.option("--cbtm_n_clusters", type=int, default=-1)
def gen_alpaca_eval_command(llama_model="gpt3", batch_size=4, model_name="", from_hf=0, model_path="", skill_selector="topic", prompt_before_lda=True, of_pref="", run_all_clusters=True, cbtm_n_clusters=-1):
    return gen_alpaca_evl(llama_model, batch_size, model_name, from_hf, model_path, skill_selector, prompt_before_lda, of_pref, run_all_clusters, cbtm_n_clusters)

def gen_alpaca_evl(llama_model="gpt3",      
         batch_size=4, model_name="", 
         from_hf=0, model_path="", 
         skill_selector="topic", 
         prompt_before_lda=True, 
         of_pref="",                          
         run_all_clusters=True, cbtm_n_clusters=-1):
    
    home=os.getenv("CONFIG_DIR","/home/v-oostapenko/dev/")
    model_dict = {
        "tloen_alpaca": {"from_hf":3, "model_name":"tloen_alpaca", "depth":0, "model_path":"none"},               
        "9psyqia3_lda_l2_topic": {"from_hf":0, "model_name":"9psyqia3_lda_l2_topic", "depth":2, "model_path":"/home/v-oostapenko/logs/amlt_yahma_llama_atlas_9psyqia3_cluster_l2/loss=0.4305.ckpt"},       
        "7ogye4hj_lora16r": {"from_hf":0, "model_name":"7ogye4hj_lora16r", "depth":2, "model_path":"/home/v-oostapenko/logs/model_from_1016/yahma_llama-7b-hf7ogye4hj_alpaca_lora_full_r16_3e_ptopt_notrainonsource_eosTrue-val/loss=0.3825.ckpt"},        
        "cxwgazou_lora4r": {"from_hf":0, "model_name":"cxwgazou_lora4r", "depth":2, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_lora_4r/alpaca_lora_4r/best_model_alpaca_lora_full_r4/loss=0.4216.ckpt"},        
        "ze3jfpzb_lda_l1_topic": {"from_hf":0, "model_name":"ze3jfpzb_lda_l1_topic", "depth":1, "model_path":"/home/v-oostapenko/logs/model_from_1016/amlt_yahma_llama_atlas_cluster_l1/alpaca4r_topic_ldal1/alpaca-lora_l1/best_model_alpaca_lora_atlas_cluster_te_ada_l1/loss=0.4242.ckpt"},
        "20massdist_m24be25e": {"from_hf":0, "model_name":"20massdist_m24be25e", "depth":1, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_wshared/alpaca_lora_20massdist/best_model_alpaca_lora_atlas_cluster_l1_20_mass_dist/loss=0.4233.ckpt"},
        "33mass_sharedad_fnp3rjlz": {"from_hf":0, "model_name":"33mass_sharedad_fnp3rjlz", "depth":1, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_wshared/alpaca_lora_shared/best_model_alpaca_lora_atlas_cluster_l1_wshared/loss=0.4227.ckpt"},     
		"4np5opiy_shared_l1_topic": {"from_hf":0, "model_name":"4np5opiy_shared_l1_topic", "depth":1, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_wshared/alpaca_lora_shared/best_model_alpaca_lora_atlas_cluster_l1_wshared/loss=0.4251.ckpt", "shared_adapter":True},		
		"pm9xjm67_lda_l3_topic": {"from_hf":0, "model_name":"pm9xjm67_lda_l3_topic", "depth":3, "model_path":"/home/v-oostapenko/logs/model_from_1016/yahma_llama-7b-hfpm9xjm67_alpaca_lora_atlas_cluster_te_ada-val/loss=0.4567.ckpt"},
        "tloen_alpaca_ae_hparams": {"from_hf":3, "model_name":"tloen_alpaca_ae_hparams", "depth":0, "model_path":"none","top_p":1.0,"max_new_tokens":2048},# "do_sample":True, "top_p":1.0,"max_new_tokens":2048},
        "chavinlo/alpaca-native": {"from_hf":1, "model_name":"chavinlo/alpaca-native", "depth":0, "model_path":"none", "top_p":1.0,"max_new_tokens":2048}, # "do_sample":True, "top_p":1.0,"max_new_tokens":2048},
        "tatsu_lab_alpaca_7b_wdiff": {"from_hf":4, "model_name":"tatsu_lab_alpaca_7b_wdiff", "depth":0, "model_path":"/home/v-oostapenko/logs/alpaca"}, # "do_sample":True, "top_p":1.0,"max_new_tokens":2048},
        # "rand_l1": {"from_hf":0, "model_name":"y3ay2zy7?_lora4r", "depth":2, "model_path":"/home/v-oostapenko/logs/model_from_1016/yahma_llama-7b-hfy3ay2zy7_alpaca_lora_atlas_cluster_te_ada-val/loss=0.4453.ckpt"},
        "ssxn3dfj_rand_l1": {"from_hf":0, "model_name":"ssxn3dfj_rand_l1", "depth":1, "model_path":"/home/v-oostapenko/logs/amlt_yahma_llama_rand_clst_l1_ssxn3dfj/randl1/alpaca_lora_randl1/best_model_alpaca_lora_atlas_cluster_l1_rand/loss=0.4353.ckpt"},
        "hard_pbuk0h30_l1": {"from_hf":0, "model_name":"hard_pbuk0h30_l1", "depth":1, "model_path":"/home/v-oostapenko/logs/alpaca_lora_hard_pbuk0h30/best_model_alpaca_lora_atlas_cluster_l1_hard/loss=0.4260.ckpt"},
        "hard_rand_jbde2wd6_l1": {"from_hf":0, "model_name":"hard_rand_jbde2wd6_l1", "depth":1, "model_path":"/home/v-oostapenko/logs/alpaca_lora_hard_rand_jbde2wd6/best_model_alpaca_lora_atlas_cluster_l1_hard_rand/loss=0.4369.ckpt"},
        "uniform_jqvvbmzh_l1": {"from_hf":0, "model_name":"uniform_jqvvbmzh_l1", "depth":1, "model_path":"/home/v-oostapenko/logs/alpaca_lora_uniform_jqvvbmzh/best_model_alpaca_lora_atlas_cluster_l1_uniform/loss=0.4222.ckpt"},
        "wizzard_alpaca_4r": {"from_hf":0, "model_name":"wizzard_alpaca_4r", "depth":1, "model_path":f"{home}/mttl/models_gcr/wizzard_loss=0.2348.ckpt"},
        "alpaca_4r_merge_after_kf2xr03j": {"from_hf":0, "model_name":"alpaca_4r_merge_after_kf2xr03j", "depth":1, "model_path":f"/home/v-oostapenko/dev/amlt/alpaca_merge_after/alpaca_lora_merge_after/best_model_alpaca_lora_atlas_cluster_l1_merge_after/loss=0.4347.ckpt"},
        "alpaca_4r_merge_after_V21ti25elu": {"from_hf":0, "model_name":"alpaca_4r_merge_after_V21ti25elu", "depth":1, "model_path":f"/home/v-oostapenko/dev/amlt/alpaca_merge_after/alpaca_lora_merge_after/best_model_alpaca_lora_atlas_cluster_l1_merge_after/loss=0.4253.ckpt"},           
        "alpaca_4r_merge_after_same_init_wg0x7gi6": {"from_hf":0, "model_name":"alpaca_4r_merge_after_same_init_wg0x7gi6", "depth":1, "model_path":f"/home/v-oostapenko/dev/amlt/alpaca_merge_after/alpaca_lora_merge_after_same_innit/best_model_alpaca_lora_atlas_cluster_l1_merge_after_same_innit/loss=0.4254.ckpt"},
        "alpaca_4r_merge_after_same_init_wd01_aen0508m": {"from_hf":0, "model_name":"alpaca_4r_merge_after_same_init_wd01_aen0508m", "depth":1, "model_path":f"/home/v-oostapenko/dev/amlt/alpaca_merge_after/alpaca_lora_merge_after_same_innitwd/best_model_alpaca_lora_atlas_cluster_l1_merge_after_same_innit_wd01/loss=0.4253.ckpt"},
        "alpaca_4r_merge_after_shared_lora_a": {"from_hf":0, "model_name":"alpaca_4r_merge_after_shared_lora_a", "depth":1, "model_path":f"/home/v-oostapenko/logs/best_model_alpaca_lora_atlas_cluster_l1_shared_loraa/loss=0.4257.ckpt"},    
        "alpaca_cbtm_dist4_ms0_temp01": {"from_hf":0, "model_name":"alpaca_cbtm_dist4_ms0_temp01", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_lora_cbtm_clustering/alpaca_cbtm_dist4_ms0_temp01/yahma_llama-7b-hfsx5tnvyw_alpaca_lora_cbtm_dist-val/best_model_alpaca_lora_cbtm_dist_loss=0.4238.ckpt/loss=0.4238.ckpt"},
        "alpaca_smear_8_pe_long_kminit": {"from_hf":0, "model_name":"alpaca_smear_8_pe_long_kminit", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_smear/alpaca_smear_8_pe_long_kminit/yahma_llama-7b-hfx39ys94j_alpaca_lora_cbtm_dense-val/loss=0.8162.ckpt"},        
        
    
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
        if "shared_adapter" in model_dict[model_name]:
            shared_adapter = model_dict[model_name]["shared_adapter"]
        
    task_results = {} 
    topic_router = None
    disable_torch_init()      
    
    model, tokenizer, config, topic_router = load_model_for_generation(from_hf, llama_model, model_path, skill_selector)
    
    
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
    out_file_name = f"{of_pref}al_eval_pred_{model_name}_{llama_model}.json"
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