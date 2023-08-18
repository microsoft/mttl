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
from inst_follow.eval.mmlu.categories import subcategories, categories
from inst_follow.utils import load_model, TopicRouter, disable_torch_init
from finetune_llama import parse_config, Config
from mttl.cluster_tuning.cluster_reader import ClusterResult
from mttl.models.poly import get_selector
from transformers import LlamaTokenizer
from peft import PeftModel
from inst_follow.models.clm import CLM
from inst_follow.eval.utils import (
    get_next_word_predictions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
)
from inst_follow.eval.gen_ni_predictions import load_model_for_generation, dict_to_dataclass
  
# from eval.model_loader import (
#     load_from_llama,
#     load_from_mttl,
#     load_from_peft,
# )


choices = ["A", "B", "C", "D"]

device = "cuda" if torch.cuda.is_available() else "cpu"


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf_model(
    args,
    subject,
    model,
    tokenizer,
    dev_df,
    test_df,
    batch_size=1,
    topic_router=None,
    skill_selector=None,
    cluster_depth=1,
):
    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        tokenized_prompt = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
        # make sure every prompt is less than 2048 tokens
        while tokenized_prompt.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            tokenized_prompt = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids

        if args.use_chat_format:
            prompt = "<|user|>\n" + prompt.strip() + "\n<|assistant|>\nThe answer is:"

        prompts.append(prompt)

    # get the answer for all examples
    # note: here we cannot directly use convert_tokens_to_ids because the some tokenizers will automatically add space prefix.
    answer_choice_ids = [
        tokenizer.encode(answer_choice, add_special_tokens=False)[0]
        for answer_choice in choices
    ]
    pred_indices, all_probs = get_next_word_predictions(
        model,
        tokenizer,
        prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=batch_size,
        topic_router=topic_router,
        skill_selector=skill_selector,
        cluster_depth=cluster_depth,
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    import tiktoken

    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [
        gpt_tokenizer.encode(" " + x)[0] for x in choices
    ]  # be careful, the tokenizer will tokenize " A" and "A" differently.

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(
        [[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))]
    )  # dummy probs, just don't want to dig into the openai probs

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

@click.command()
@click.option("--ntrain", type=int, default=5)
@click.option("--data_dir", type=str, default="data/mmlu")
@click.option("--save_dir", type=str, default="results/mmlu/llama-7B/")
@click.option(
    "--model_name",
    type=str,
    default="alpaca_smear_12_xr4_cos",
    help="if specified, we will load the model to generate the predictions.",
)

@click.option(
    "--model_path",
    type=str,    
    default='/home/v-oostapenko/dev/amlt//alpaca_smear/alpaca_smear_12_xr4_cos/yahma_llama-7b-hfnekcmtyy_alpaca_lora_cbtm_dense-val/loss=0.8796.ckpt',
    help="if specified, we will load the model to generate the predictions.",
)
@click.option(   
    "--openai_engine",
    type=str,
    default=None,
    help="if specified, we will use the OpenAI API to generate the predictions.",
)
@click.option(
    "--n_instances",
    type=int,
    default=None,
    help="if specified, a maximum of n_instances per subject will be used for the evaluation.",
)
@click.option(
    "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
)
@click.option(
    "--skill_selector",
    type=str,
    default="poly", 
    help="skill selector",
)
@click.option("--amlt_experiment_name", type=str, default="alpaca_smear")
@click.option("--use_chat_format", is_flag=False)
@click.option("--subjects", type=str, default="-1")
def main(ntrain=5,     
         data_dir="/home/v-oostapenko/data/mmlu", 
         save_dir="/home/v-oostapenko/results/mmlu/llama-7B/", 
         model_name="",
         model_path="", 
         #tokenizer_name_or_path, 
         openai_engine=None,
         subjects=-1, 
         n_instances=None, 
         eval_batch_size=1,         
         skill_selector="topic",        
         amlt_experiment_name="alpaca_smear", 
         use_chat_format=False):
    return eval_mlu(ntrain, data_dir, save_dir, model_name, model_path, openai_engine, subjects, n_instances, eval_batch_size, skill_selector, amlt_experiment_name, use_chat_format)

def eval_mlu(ntrain=5,     
             data_dir="/home/v-oostapenko/data/mmlu", 
             save_dir="/home/v-oostapenko/results/mmlu/llama-7B/", 
             model_name="",
             model_path="", 
             #tokenizer_name_or_path, 
             openai_engine=None,
             subjects=-1, 
             n_instances=None, 
             eval_batch_size=1,         
             skill_selector="topic",        
             amlt_experiment_name="alpaca_smear", 
             use_chat_format=False):
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
    
    # pack all the arguments into dict
    args = {"ntrain": ntrain, "data_dir": data_dir, "save_dir": save_dir, "model_name": model_name, "model_path": model_path, 
            "openai_engine": openai_engine, "subjects": subjects, "n_instances": n_instances, 
            "eval_batch_size": eval_batch_size, "skill_selector": skill_selector, "amlt_experiment_name": amlt_experiment_name, 
            "use_chat_format": use_chat_format, "from_hf": from_hf}
    args = dict_to_dataclass(args)
    
    
    base_model = "yahma/llama-7b-hf"             
    model, tokenizer, config, topic_router = load_model_for_generation(from_hf, base_model, 
                                                                       model_name, 
                                                                       model_path, 
                                                                       skill_selector)
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if subjects:
        assert all(
            subj in subjects for subj in subjects
        ), f"Some of the subjects you specified are not valid: {subjects}"
        subjects = subjects

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(os.path.join(save_dir))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )
        if n_instances and n_instances < test_df.shape[0]:
            test_df = test_df.sample(n_instances, random_state=42)

        # if mode:
        # if skill_selector == "topic":
        #     cors, acc, probs = eval_hf_model(
        #         args,
        #         subject,
        #         model,
        #         tokenizer,
        #         dev_df,
        #         test_df,
        #         eval_batch_size,
        #         topic_router=topic_router,
        #         skill_selector=skill_selector,
        #         cluster_depth=cluster_depth,
        #     )
        # else:
        
        cors, acc, probs = eval_hf_model(
            args,
            subject,
            model,
            tokenizer,
            dev_df,
            test_df,
            eval_batch_size,
        )
        # else:
        #     cors, acc, probs = eval_openai_chat_engine(
        #         args, subject, openai_engine, dev_df, test_df, eval_batch_size
        #     )

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["choice{}_probs".format(choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(save_dir, "{}.csv".format(subject)),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # save results
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat])) for cat in cat_cors
                },
            },
            f,
        )
    return weighted_acc

        
if __name__ == "__main__":
    main()
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ntrain", type=int, default=5)
#     parser.add_argument("--data_dir", type=str, default="data/mmlu")
#     parser.add_argument("--save_dir", type=str, default="results/mmlu/llama-7B/")
#     parser.add_argument(
#         "--model_name_or_path",
#         type=str,
#         default=None,
#         help="if specified, we will load the model to generate the predictions.",
#     )
#     parser.add_argument(
#         "--tokenizer_name_or_path",
#         type=str,
#         default=None,
#         help="if specified, we will load the tokenizer from here.",
#     )
#     parser.add_argument(
#         "--openai_engine",
#         type=str,
#         default=None,
#         help="if specified, we will use the OpenAI API to generate the predictions.",
#     )
#     parser.add_argument(
#         "--subjects",
#         nargs="*",
#         help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated.",
#     )
#     parser.add_argument(
#         "--n_instances",
#         type=int,
#         help="if specified, a maximum of n_instances per subject will be used for the evaluation.",
#     )
#     parser.add_argument(
#         "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
#     )
#     parser.add_argument(
#         "--load_in_8bit",
#         action="store_true",
#         help="load model in 8bit mode, which will reduce memory and speed up inference.",
#     )
#     parser.add_argument(
#         "--gptq",
#         action="store_true",
#         help="If given, we're evaluating a 4-bit quantized GPTQ model.",
#     )
#     parser.add_argument(
#         "--use_chat_format",
#         action="store_true",
#         help="If given, the prompt will be encoded as a chat format with the roles in prompt.",
#     )
#     parser.add_argument(
#         "--example_to_ids_path",
#         type=str,
#         default="inst_follow/cluster_infos/atlas_by_instr_bert-base-uncased_ldalayer2.pkl",
#         help="path to the example_to_ids file.",
#     )
#     parser.add_argument(
#         "--skill_selector",
#         type=str,
#         default="poly",
#         help="skill selector",
#     )
#     parser.add_argument(
#         "--load_from",
#         type=str,
#         default="mttl",
#         help="source to load the model from",
#     )
#     parser.add_argument(
#         "--cluster_depth",
#         type=int,
#         default=1,
#         help="cluster depth",
#     )
#     args = parser.parse_args()

#     # model_name_or_path and openai_engine cannot be both None or both not None.
#     assert (model_name_or_path is None) != (
#         openai_engine is None
#     ), "Either model_name_or_path or openai_engine should be specified."
#     main(args)