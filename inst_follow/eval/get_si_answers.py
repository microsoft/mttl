# from 
import os
import sys
import ray
import copy
import json 
import time
import torch
# import click
import openai
from tqdm import tqdm
import numpy as np
import argparse
from typing import List      
from inst_follow.models.clm import CLM  
from mttl.models.modify_model import modify_transformer  
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.dataloader.data_utils import ExampleInfo
from inst_follow.finetune_llama import parse_config, Config
from fastchat.serve.inference import compute_skip_echo_len
from mttl.cluster_tuning.cluster_reader import ClusterResult
from inst_follow.utils import load_model,TopicRouter
from transformers import AutoTokenizer, AutoModelForCausalLM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def open_ai_request(prompt):
        try: 
            response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",          
                    messages=[{"role": "user", "content": f"{prompt}"}])
        except:
            print("Error in generating task response")
            # retrying for 5 times
            for i in range(5):                
                time.sleep(2)
                try:
                    response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",          
                            messages=[{"role": "user", "content": f"{prompt}"}])
                    break
                except Exception as e:
                    print(e)
                    if i == 4:
                        print("Failed to generate task response")
                        raise e
        return response

@ray.remote(num_cpus=4)  
def ask_openai(input):                         
    prompt = f"{input}\
                \n Remember, you must only generate the response without any additional comments. Start you answer with ###My_Answer:."
    response = open_ai_request(prompt)
    summary_word = response.choices[0].message.content
    try:
        answer = summary_word.split("My_Answer:")[1].strip()
    except:
        answer = "Error in generating answer"
    return answer


@torch.no_grad()
def generate_outputs_by_one(args, model, tokenizer, cluster_result, examples:List[ExampleInfo], adapter_id, max_output_length=256):
    otuputs_list=[]   
    
    if args.model_name == "openai":  
        handles=[]
        for example in examples:   
            line:ExampleInfo = example
            qs = line.input_text
            example_hash = line.hash
            # instruction = qs.split("\n### Instruction:")[1].split("\n### Response:")[0]
            #remove response              
            input = qs.split("\n### Response:")[0] # remove the response from the input         
            handles.append(ask_openai.remote(input))    
            # otuputs_list.append(ask_openai(input))
        otuputs_list=ray.get(handles)
        return otuputs_list
    
    
    for example in examples:   
        line:ExampleInfo = example
        qs = line.input_text
        example_hash = line.hash
        # instruction = qs.split("\n### Instruction:")[1].split("\n### Response:")[0]
        #remove response              
        input_text = qs.split("\n### Response:")[0] # remove the response from the input     
        input_text+="\n### Response:"                  
        inputs = tokenizer([input_text]) # using tokenizer defined for generation, not the one from dm (which appends eos, that we dont need here)

        
        model.model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.model.config.bos_token_id = 1
        model.model.config.eos_token_id = 2       
        probs = cluster_result.get_distances(example_hash)
        #set adapter_id
        if adapter_id is not None:
            probs = np.zeros_like(probs)
            probs[adapter_id] = 1
            #set adapter_id
        input={                         
            "input_ids": torch.as_tensor(inputs.input_ids).cuda(),
            "task_ids": torch.zeros(1, dtype=torch.long).cuda()*-1,
        }        
        if isinstance(probs, List):
            probs = np.array(probs)
        input["distances"] = probs[None,...]       
        # eval with the best adapter for this cluster
        output_ids = model.generate(
            input,
            # do_sample=True,
            temperature=0.7,       
            max_new_tokens=max_output_length,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        output_cleaned = outputs.split(input_text)[1]           
        otuputs_list.append(output_cleaned)
    return otuputs_list


@torch.no_grad()
def generate_outputs(args, model, tokenizer, cluster_result, examples:List[ExampleInfo], adapter_id, max_output_length=1256):
    otuputs_list=[]   
    
    if args.model_name == "openai":  
        handles=[]
        for example in examples:   
            line:ExampleInfo = example
            qs = line.input_text
            example_hash = line.hash
            # instruction = qs.split("\n### Instruction:")[1].split("\n### Response:")[0]
            #remove response              
            input = qs.split("\n### Response:")[0] # remove the response from the input         
            handles.append(ask_openai.remote(input))    
            # otuputs_list.append(ask_openai(input))
        otuputs_list=ray.get(handles)
        return otuputs_list
    
    prompts=[]
    assignment_probs=[]
    for example in examples:   
        line:ExampleInfo = example
        qs = line.input_text
        example_hash = line.hash           
        if args.prompt_type=="SI":
            input_text = qs.split("\n### Response:")[0] # remove the response from the input     
            input_text+="\n### Response:"  
        elif args.prompt_type=="custom":
            raise NotImplementedError("Custom prompt not implemented")
            # input_text = qs.split("\n### Instruction:")[1].split("\n### Response:")[0] # remove the response from the input
            # input_text = "\n### Instruction:"+input_text
            # input_text+="\n### Response:"  
            # input_text 
        probs = cluster_result.get_distances(example_hash)                
        if adapter_id is not None:
            probs = np.zeros_like(probs)
            probs[adapter_id] = 1
        if isinstance(probs, List):
            probs = np.array(probs)
        prompts.append(input_text)   
        assignment_probs.append(probs) 
            
    inputs = tokenizer(prompts,
        padding=True,
        return_tensors="pt")   
    batch_size=8
    for i in range(0, len(examples), batch_size): 
        prompts_batch = prompts[i:i+batch_size]         
        input_ids_batch = inputs.input_ids[i:i+batch_size].cuda()
        assignment_probs_b = assignment_probs[i:i+batch_size]
        model.model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.model.config.bos_token_id = 1
        model.model.config.eos_token_id = 2       
        input={                         
            "input_ids": input_ids_batch.cuda(),
            "task_ids": torch.zeros(len(input_ids_batch), dtype=torch.long).cuda()*-1,
        }        
        if isinstance(assignment_probs_b, List):
            probs = np.stack(assignment_probs_b)
            input["distances"] = probs
            
        output_ids = model.generate(
            input,
            # do_sample=True,
            temperature=0.7,       
            max_new_tokens=max_output_length,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # output_cleaned = outputs.split(input_text)[1]                  
        output_cleaned = [out.split(ex)[1] for out,ex in zip(outputs, prompts_batch)]  
        otuputs_list.extend(output_cleaned)
    return otuputs_list

def main(args):
    rng = np.random.RandomState(args.seed)
    cluster_result = None
    if args.example_to_ids_path is not None:
        cluster_result = ClusterResult(args.example_to_ids_path)
        topic_router = TopicRouter(args.cluster_with)
    all_topics=topic_router.map.get_topic_data()                               
    cluster_ids = [np.argmax(dists) for dists in cluster_result.infos.cluster_dists] # could also use cluster_result.infos.cluster_ids (51760 points), these are ids of the topics that are most likely for each datapoint. 
    #This also corresponds to the id of tha adapter used in poly_lora (lora_a and lora_b).
    assert cluster_ids == cluster_result.infos.cluster_ids

    counts_per_topic=np.bincount(cluster_ids) # there are some topics that are never used as a primary topic. Why? Its hirarchical clustering, hence some topics are used at other clustering layers (higher hirarchy). 
    # more about it, here https://github.com/nomic-ai/nomic/blob/5a90f6a30dbce946d1b5a616e8d3dcc1e1008097/docs/how_does_atlas_work.md?plain=1#L59
    
    valid_topics = [i for i, c in enumerate(counts_per_topic) if c > 100]  
    random_topics = rng.choice(valid_topics, args.n_topics, replace=False)  
    selected_topics = [64,122,119,228,111,183,221,166] # additional topics that we might want to test
    topics = np.concatenate([selected_topics, random_topics])
    topics = np.unique(topics)
    
    print(f"Selected {len(topics)} topics: {topics}")   
    
    # for each topic, select some examples
    example_hashes_all_topics=[]
    for t in topics:
        topic_hashes=[]       
        for i, dists in enumerate(cluster_result.infos.cluster_dists):
            if (np.argmax(dists) == t and cluster_result.infos.is_test[i]==0):
                topic_hashes.append(cluster_result.infos.hashes[i]) 
        example_hashes_all_topics.append(rng.choice(topic_hashes, args.n_examples_per_topic, replace=False))
    assert len(example_hashes_all_topics) == len(topics)
    
    if os.path.exists(args.destination):
        with open(args.destination, "r") as f:
            lines = f.readlines()
        last_line = lines[-1]    
        last_topic_id = json.loads(last_line)["topic_id_examples"]
        last_topic_id_idx = np.where(topics == int(last_topic_id))[0][0]
        example_hashes_all_topics = example_hashes_all_topics[last_topic_id_idx+1:]
        topics = topics[last_topic_id_idx+1:]    
    print(topics, len(example_hashes_all_topics))
    
    config = Config()
    config.model = "yahma/llama-7b-hf" # used to load tokenizer in AlpacaDataModule, we dont care about it here
    dm = AlpacaDataModule(config) 
    dst = dm.get_dataset()              
     
    print("Loading model") 
    if args.hash_to_idx_path is not None:
        with open(args.hash_to_idx_path, "r") as f:
            hash_to_idx = json.load(f)
    else:
        hash_to_idx = {ex.hash: i for i, ex in tqdm(enumerate(dst))} # building a sample-hash to index in the dst map for more speed        
        
        
    if args.model_name=="openai":
        model, tokenizer = None, None
        print("Will ask openai for responses")
    else:
        config.model = args.model_name
        config.n_skills = 1 # by default, can be overwritten in load_model if a checkpoint is provided
        model, tokenizer = load_model(config, args.model_path)  
        print(f"Loaded model {args.model_name} from {args.model_path}\n")
        print("Loaded config", config.__dict__)
        if model.args.n_skills>1:
            assert cluster_result is not None, "For soft-clustering models, cluster_result must be provided"
            assert model.args.n_skills == len(all_topics) == cluster_result.n_clusters()   
        
            
        if args.usepijma_model_with_llama_adapter:    
            model.to("cpu")          
            pijama_model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1", torch_dtype=torch.float32)#, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1")#.to(device)
            tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
            tokenizer.padding_side='left'            
            args_pij = copy.deepcopy(config) #{"adapter_modules": "attention", "model_modifier":"llama_adapter"} 
            args_pij.model="togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
            # args = dict_to_dataclass(args) 
            args_pij.adapter_modules = "attention"
            pijama_model = modify_transformer(pijama_model, args_pij)
            # pijama_model.to("cpu")
            # args.lora_modules="v_proj|q_proj|k_proj|o_proj" #"attention"
            state_dict = model.model.state_dict()
            #load adaption_prompt and adaptation_gate weights into args.model_object
            new_state_dict = {}       
            for k,n in state_dict.items():   
                if "adaption_prompt" in k or "adaption_gate" in k:
                    #rename key      
                    ad_layer = k.split(".")[-1]
                    layer = k.split("layers.")[1].split(".")[0]
                    # layers.2.attention.adaption_prompt
                    k = f"gpt_neox.layers.{layer}.attention.{ad_layer}"
                    new_state_dict[k] = n
            pijama_model.load_state_dict(new_state_dict, strict=False)    
            assert  int(sum([torch.sum(p) for n,p in model.model.named_parameters() if "adaption" in n]).item()) == int(sum([torch.sum(p) for n,p in pijama_model.named_parameters() if "adaption" in n]).item())
            # pijama_model.to(device)
            # model.to(device)     
            model_class = CLM   
            args_pij.model_object = pijama_model 
            # tokenizer = dm.tokenizer if dm is not None else tokenizer
            model = model_class(**vars(args_pij), tokenizer=tokenizer)
            model.to(device)
            model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
            model.model.config.bos_token_id = tokenizer.bos_token_id
            model.model.config.eos_token_id = tokenizer.eos_token_id  
        
            
            
    
    topics_for_adapters = topics
    if hasattr(tokenizer, "add_eos_token"):
        assert tokenizer.add_eos_token == False
    if args.batched:
        tokenizer.padding_side = "left" # to nebale atched generation  
    print("Starting generating the answers")  
    for t,examples in enumerate(example_hashes_all_topics):    
        otuputs_list=[] 
        examples_hashes=examples.tolist()
        examples_data = [dst[hash_to_idx[h]] for h in examples_hashes]
        print(f"Testing with topic {all_topics[topics[t]]['topic_description']}, with {len(examples_data)} examples")
        # test on the correct mixture of adapters
        adapter_id = None 
        print( f"        Tesing with correct mixture of adapters from the examples from |{all_topics[topics[t]]['topic_description']}|")
        if args.batched:
            output = generate_outputs(args,model,tokenizer,cluster_result,examples_data, adapter_id)
        else:
            output = generate_outputs_by_one(args,model,tokenizer,cluster_result,examples_data, adapter_id)
        otuputs_list.append(output)
        # test on seperate adapters
        adapter_ids = [("gt", "mixture")] if model.args.n_skills>1 else [("full_model", "full_model")]
        if config.n_skills > 1:
            for adapter_id in topics_for_adapters:  
                adapter_ids.append((adapter_id, all_topics[adapter_id]['topic_description']))
                print( f"        Tesing with adapter {adapter_id} from |{all_topics[adapter_id]['topic_description']}| on examples from |{all_topics[topics[t]]['topic_description']}|")
                if args.batched:
                    output = generate_outputs(args,model,tokenizer,cluster_result,examples_data, adapter_id)
                else:
                    output = generate_outputs_by_one(args,model,tokenizer,cluster_result,examples_data, adapter_id)
                otuputs_list.append(output)      
        # write result into a file: in total these are len(topics)+1*len(examples_data) lines
        # store it in a json file, with the following format:
        # {"topic_id_examples": ,"topic_descr_examples":  ,"adapter_id":  , "adapter_descr":  ,"hash":  ,"query:" ,"responses":"}
        with open(args.destination, "a") as f:
            for i, output in enumerate(otuputs_list): # output for each of the 11 adapters
                for j in range(len(examples_data)): # output for each of the len(examples_data) examples
                    out_example = output[j]
                    #append line to a file
                    f.write(json.dumps({           
                        "topic_id_examples": str(topics[t]),  
                        "topic_descr_examples": all_topics[topics[t]]['topic_description'],
                        "adapter_id": str(adapter_ids[i][0]),
                        "adapter_descr": str(adapter_ids[i][1]),
                        "hash": examples_data[j].hash,
                        "query": examples_data[j].input_text.split("\n### Response:")[0],#.split("\n### Instruction:")[1],
                        "response": out_example,
                    })+"\n")

if __name__ == "__main__":          
    parser = argparse.ArgumentParser(add_help=False)        
    # parser.add_argument("--model-id",type=str, default="alpaca")
    parser.add_argument("--model_name", type=str, default="yahma/llama-7b-hf") # for own alpaca yahma/llama-7b-hf + model-path, for openai "openai", for some other hf model, the name of the model and model-path None, alpaca_full chavinlo/alpaca-native and no model-path
    parser.add_argument("--model-path", type=str, default="/home/v-oostapenko/logs/llama_alpaca/llama_adapter/yahma_llama-7b-hfkrd9zi2k_llama_adapter-val/loss=0.6232.ckpt") #/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt")#"/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt")   
    parser.add_argument("--prompt_type", type=str, default="SI", choices=["SI", "custom"])
    parser.add_argument("--cluster_with", type=str, default="instruction", choices=["instruction", "response"])
    parser.add_argument("--embeddings_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/self_instruct_GPT3")
    parser.add_argument("--destination", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/self_instruct/good_stuff/alpaca_soft_c_100x30.jsonl") #/home/v-oostapenko/dev/mttl/compositional_adapters/eval/table/answer/open_ai_gpt-3.5-turbo_responses.jsonl")
    parser.add_argument("--example_to_ids_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002.pkl") 
    parser.add_argument("--n_examples_per_topic", type=int, default=30)
    parser.add_argument("--n_topics", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batched", type=int, default=0)     
    parser.add_argument("--usepijma_model_with_llama_adapter", type=int, default=1) 
    parser.add_argument("--hash_to_idx_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/hash_to_idx_si.json") # only for debugging
    args = parser.parse_args()
    main(args)#, config)