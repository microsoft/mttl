# from   
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
import torch
import os
import ray
import sys
import json 
import time
import shortuuid
import numpy as np 
from tqdm import tqdm  
from inst_follow.finetune_llama import parse_config
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from if.gpt_review_utils import gen_prompt, get_json_list,get_eval,parse_score,parse_score_with_expert

MAX_API_RETRY = 5
REQ_TIME_GAP = 10

def load_answers(args):
    """
    Returns a dictionary of answers for each model answers_file in args.answers_files, ground truth answers are loaded from args.gt_answers (if file exists) and hash_to_idx.
    hash_to_idx - a dictionary mapping hash to index in the dst, there dst is either the alpaca dataset from AlpacaDataModule (if args.gt_answers=="ground_truth") or a list of jsons loaded from args.gt_answers
    """        
    # load the candidate answers    
    file_main = args.answers_files[args.main_answers_file_idx]
    with open(file_main, "r") as f:
        main_cand_answers = [json.loads(line) for line in f.readlines()]
    candidate_answers = {}
    for file in args.answers_files[args.main_answers_file_idx+1:]:
        with open(file, "r") as f:
            answers_list = [json.loads(line) for line in f.readlines()]
            candidate_answers[file] = answers_list
    return main_cand_answers, candidate_answers
    
def main(args):
    # load json         
    main_cand_answers, candidate_answers = load_answers(args)    
    # destination_scores = args.answers_file.split(".jsonl")[0]+f"_scores_gt:[{gt_name}].jsonl"
    # destination_scores.replace("/","_")
    
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)            
    review_jsons = []      
    
    
    # check that we have same topics in all answers files
    
    
    # all_topic_ids = np.unique([a["topic_id_examples"] for a in all_candidate_answers])
    
    # #restart from last generated topic
    # # read the output file
    # if os.path.exists(destination_scores):
    #     with open(destination_scores, "r") as f:
    #         lines = f.readlines()
    #     last_line = lines[-1]    
    #     last_topic_id = json.loads(last_line)["topic_id_examples"]
    #     last_topic_id_idx = np.where(all_topic_ids == last_topic_id)[0][0]
    #     all_topic_ids = all_topic_ids[last_topic_id_idx+1:]    
    # print(all_topic_ids)
    
    
    
    for topic_id in tqdm(all_topic_ids):
        all_points_topic = [p for p in answers_list if p["topic_id_examples"] == topic_id]
        # gt_examples = [a for a in all_points_topic if a["adapter_id"] == "gt"]
        handles = []
        exmples_evaluated = []
        for example in all_points_topic:
            query, response, hash, category = example["query"], example["response"], example["hash"], example["topic_descr_examples"]
            # all_other_examples = [a for a in all_points_topic if a["adapter_id"] != "gt"]
            
            # handles = []
            # exmples_evaluated = []
            # for other_example in all_other_examples:
            #     if other_example["hash"] == hash:
                    # print(f"Evaluating query {query} with response {response} on adapter {other_example['adapter_id']} with response {other_example['response']}")
            # example_response=example["response"]
            if isinstance(dst, list):
                ans_gt = dst[hash_to_idx[hash]]["response"]
                assert dst[hash_to_idx[hash]]["query"]==query
            else:
                assert model_1_name == "ground_truth"
                ans_gt = dst[hash_to_idx[hash]]["input_text"].split("Response:")[1].strip()
            ans2 = example["response"]
            ques = query
            cat = "general"
            if model_1_name == "ground_truth":
                cat+="_expert"                  
            sys_prompt, prompt, reviewer_id = gen_prompt(reviewer_jsons,
                                                        prompt_jsons, 
                                                        cat, 
                                                        ques, 
                                                        ans_gt, 
                                                        ans2)
            review_id = shortuuid.uuid()
            review_jsons.append({
                'review_id': review_id,
                'reviewer_id': reviewer_id,
                'metadata': {},
            })
            # To avoid the rate limit set by OpenAI
            handles.append(get_eval.remote(sys_prompt, prompt, 1024))
            time.sleep(REQ_TIME_GAP)
            exmples_evaluated.append(example)
                    
        reviews = ray.get(handles)
        scores_all = []
        for idx, review in enumerate(reviews):
            if model_1_name == "ground_truth":
                score = parse_score_with_expert(review)
            else:
                score = parse_score(review)
            scores_all.append(score)
            
        with open(destination_scores, "a") as f:
            # #write ground truth
            # gt_example["score"] = [10.0, 10.0]
            # f.write(json.dumps(gt_example)+"\n")      
            for i, output in enumerate(exmples_evaluated): # output for each of the 11 adapters
                out_example = output
                out_example[f"score_gt:[{model_1_name}]"] = scores_all[i]
                #append line to a file
                f.write(json.dumps(out_example)+"\n")
                    

if __name__ == "__main__":          
    parser = argparse.ArgumentParser(add_help=False)       
    parser.add_argument("--prompt_file", type=str, default="/home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/prompt.jsonl")
    parser.add_argument("--reviewer_file", type=str, default="/home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/reviewer.jsonl")    
    parser.add_argument("--answer_files", nargs="+", type=str, default=[])   
    parser.add_argument("--main_answers_file_idx", type=int, default=0)
    
    args = parser.parse_args()
    main(args)