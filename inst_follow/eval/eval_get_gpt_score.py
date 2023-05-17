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
from mttl.dataloader import ni_metrics 
from inst_follow.finetune_llama import parse_config
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from inst_follow.utils import gen_prompt, get_json_list,get_eval,parse_score,parse_score_with_expert

MAX_API_RETRY = 5
REQ_TIME_GAP = 10


def main(args):
    # load json       
    config, args = parse_config(parent=parser, return_parser=True)   
    config.model= "yahma/llama-7b-hf"
    # model_2_name = args.answers_file.split("/")[-1].split(".jsonl")[0]
    dst=None
    if args.gt_answers=="none":
        model_1_name="none"
    elif args.gt_answers=="ground_truth":
        model_1_name = "ground_truth"
        dm = AlpacaDataModule(config) 
        dst = dm.get_dataset()
        # hash_to_idx = {ex.hash: i for i, ex in tqdm(enumerate(dst))} # building a sample-hash to index in the dst map for more speed
        
        if args.hash_to_idx_path is not None:  
            with open(args.hash_to_idx_path, "r") as f:
                    hash_to_idx = json.load(f)
        else:                                                 
            hash_to_idx = {ex.hash: i for i, ex in tqdm(enumerate(dst))}
    else:
        model_1_name = args.gt_answers.split("/")[-1].split(".json")[0]
        # load responses model 1
        with open(args.gt_answers, "r") as f:
            dst = [json.loads(line) for line in f.readlines()]
        hash_to_idx = {ex["hash"]: i for i, ex in tqdm(enumerate(dst))} #
      
    destination_scores = args.answers_file.split(".jsonl")[0]+f"_sc_{args.score}_vs_gt{model_1_name}_{args.response_field}.jsonl"
    destination_scores.replace("/","_")
    
    with open(args.answers_file, "r") as f:
        answers_list = [json.loads(line) for line in f.readlines()]
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)
            
    review_jsons = []      
    all_topic_ids = np.unique([a["topic_id_examples"] for a in answers_list])
    
    # restart from last generated topic
    # read the output file
    if os.path.exists(destination_scores):
        with open(destination_scores, "r") as f:
            lines = f.readlines()
        last_line = lines[-1]    
        last_topic_id = json.loads(last_line)["topic_id_examples"]
        last_topic_id_idx = np.where(all_topic_ids == last_topic_id)[0][0]
        all_topic_ids = all_topic_ids[last_topic_id_idx+1:]
    
    print(all_topic_ids)
    for topic_id in tqdm(all_topic_ids):
        print(topic_id)
        scores_all = []
        all_points_topic = [p for p in answers_list if p["topic_id_examples"] == topic_id]
        # gt_examples = [a for a in all_points_topic if a["adapter_id"] == "gt"]
        handles = []
        exmples_evaluated = []
        answs_gt=[]
        ans_pred = []
        gt_responses=[]
        for example in all_points_topic:
            query, response, hash, category = example["query"], example["response"], example["hash"], example["topic_descr_examples"]
            ans_gt=None
            if isinstance(dst, list):
                ans_gt = dst[hash_to_idx[hash]]["response"]
                assert dst[hash_to_idx[hash]]["query"]==query
            elif model_1_name == "ground_truth":   
                ans_gt = dst[hash_to_idx[hash]].input_text.split("Response:")[1].strip()                
            
            ans2 = example[args.response_field]
            # remove the SI prompt!
            ques = query
            ques = "\n### Instruction:"+query.split("\n### Instruction:")[1]
            cat = "general"
            if ans_gt is not None:
                gt_responses.append(ans_gt)  
            
            if model_1_name == "ground_truth":
                cat+="_against_expert"    
            if args.gt_answers=="none":
                cat+="_agains_none"   
                if "\n### Input:" in ques:
                    cat+="_w_input"
            if args.score == "GPT":                       
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
            else:
                rougeL = ni_metrics.compute_metrics([ans2],[[ans_gt]])['rougeL']
                scores_all.append(rougeL)
                exmples_evaluated.append(example)
        if args.score == "GPT":      
            reviews = ray.get(handles)
            # scores_all = []
            all_reviews=reviews
            for idx, review in enumerate(reviews):
                if model_1_name == "ground_truth" or args.gt_answers=="none": 
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
                if f"score_{args.score}_against:[{model_1_name},{args.response_field}]" not in out_example:                       
                    out_example[f"score_{args.score}_against:[{model_1_name},{args.response_field}]"] = scores_all[i]
                if "gt_response" not in out_example and len(gt_responses)>0:
                    out_example["gt_response"] = gt_responses[i]
                if args.score == "GPT":   
                    if f"review_{args.score}_against:[{model_1_name},{args.response_field}]" not in out_example:
                       out_example[f"review_{args.score}_against:[{model_1_name},{args.response_field}]"] = all_reviews[i]             
                #append line to a file
                f.write(json.dumps(out_example)+"\n")
                    

if __name__ == "__main__":          
    parser = argparse.ArgumentParser(add_help=False)            
    parser.add_argument("--prompt_file", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/table/prompt.jsonl")
    parser.add_argument("--reviewer_file", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/table/reviewer.jsonl")
    
    parser.add_argument("--answers_file", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answers_alpaca_lora_full_7b4r.jsonl") #self_instruct/good_stuff/missing_answers_alpaca_sc_GPT_vs_gtground_truth_response.jsonl") #answers_llama7b_on_si_train[si_prpt][sc_GPT_vs_ground_truth][sc_rouge_vs_ground_truth][cleaned_response].jsonl") #alpaca7b_lorar4_sc_atlas_on_si_train18[si_prpt][sc_GPT_vs_ground_truth][sc_rouge_vs_ground_truth].jsonl") #answers_llama7b_on_si_train[si_prpt][sc_GPT_vs_ground_truth][sc_rouge_vs_ground_truth][cleaned_response]_sc_GPT_vs_gtground_truth_response_cleaned.jsonl") # answers_llama7b_on_si_train[si_prpt][sc_GPT_vs_ground_truth][sc_rouge_vs_ground_truth][cleaned_response]_sc_GPT_vs_gtground_truth_response_cleaned_corrected answers_llama7b_on_si_train[si_prpt][sc_GPT_vs_ground_truth][sc_rouge_vs_ground_truth][cleaned_response]_sc_GPT_vs_gtground_truth_response_cleaned_corrected.jsonl") #answers_llama7b_on_si_train[si_prpt][sc_GPT_vs_ground_truth][sc_rouge_vs_ground_truth][cleaned_response].jsonl")    
    parser.add_argument("--gt_answers", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answer_gpt35.jsonl") #ground_truth") #ground_truth")
    parser.add_argument("--score", type=str, default="GPT", choices=["GPT", "rouge"])
    parser.add_argument("--response_field", type=str, default="text")#response
    parser.add_argument("--hash_to_idx_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/hash_to_idx_si.json")
    
    
    args = parser.parse_args()
    main(args)