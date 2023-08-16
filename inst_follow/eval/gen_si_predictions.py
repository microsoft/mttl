import json
import os 
import numpy as np
import click
import json
import tqdm
import copy
import torch  
import sys
from datasets import load_dataset
# append parent dir to sys.path
# sys.path.append("..")
# sys.path.append("/home/v-oostapenko/dev/mttl")    
# sys.path.append(os.getenv("CONFIG_DIR", "/home/v-oostapenko/dev/mttl"))

directory = os.getenv("CODE_DIR", "/home/v-oostapenko/dev/mttl")
sys.path.append(directory)
sys.path.append("../..") 
sys.path.append("/mnt/amlt_code/")
print(sys.path)
print("getdefaultencoding: ",sys.getdefaultencoding())

# print the output of system command
# print(os.popen('ls -l /mnt/').read())
# print(os.popen('ls -l /').read())
# print(os.popen('ls -l /home').read())
print(os.popen('ls -l /mnt/default/data').read())    
print(os.popen('ls -l /mnt/default/data/natural-instructions/tasks').read())
   
from inst_follow.models.clm import CLM  
from mttl.dataloader import ni_metrics   
from transformers import LlamaTokenizer  
from mttl.models.poly import get_selector           
from mttl.models.modify_model import modify_transformer  
from finetune_llama import parse_config, Config
from inst_follow.utils import load_model, TopicRouter,disable_torch_init
from mttl.cluster_tuning.cluster_reader import ClusterResult
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
def dict_to_dataclass(d):
    from dataclasses import make_dataclass
    return make_dataclass("X", d.keys())(**d)

@torch.no_grad() 
def generate_outputs(model, examples, tokenizer, temperature=0.7, max_output_length=128, topic_router=None, skill_selector="topic"):
    otuputs_list=[]               
    inputs = tokenizer(examples,
            padding='longest',
            return_tensors="pt")    
    input={                         
        "input_ids": inputs.input_ids.to(device),
        "task_ids": torch.zeros(len(examples), dtype=torch.long).to(device)*-1,
    }           
    if topic_router:      
        if skill_selector=="random":
            # random binary matrix 
            raise NotImplementedError()
            input["distances"] = torch.randint(0,2,(len(examples), topic_router.n_skills)).cuda()
        else: 
            probs = topic_router(examples, depth=2)   
            if hasattr(model, "skill_ids_to_keep"):
                probs = probs[:,model.skill_ids_to_keep]
            input["distances"] = probs
                
        
    # eval with the best adapter for this cluster
    output_ids = model.generate(
        input,
        # do_sample=True,
        temperature=temperature,      
        max_new_tokens=max_output_length,
        # top_k=50,
        return_dict_in_generate=True,
    )
    # for out,ex in zip(output_ids.sequences, examples):
    #     # out = out[len(ex):] 
    #     output_str = tokenizer.decode(out)
    #     otuputs_list.append(output_str)
    outputs = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)   
    examples_in_decoded = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True) # just in case, to make sure inputs are exactly the same as expected here when spliting the decodings
    for out,ex in zip(outputs, examples_in_decoded):
        o = out.split(ex)
        assert len(o)>1 
        otuputs_list.append(o[1])
    # output_cleaned = [out.split(ex)[1] for out,ex in zip(outputs, examples)]           
    # otuputs_list.extend(output_cleaned)
    del output_ids
    del input  
    return otuputs_list
       
def format(inst:dict, examples:list):      
    out="Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."  
    for example in examples:
        out += f"\n### Instruction: {example['instruction']}"
        if len(example["input"])>0:
            out += f"\n### Input: {example['input']}"
        out += f"\n### Response: {example['output']}"
    out += f"\n### Instruction: {inst['instruction']}"
    if len(inst["input"])>0:
        out += f"\n### Input: {inst['input']}"
    out += f"\n### Response:"
    return out
          
@click.command()                   
# @click.option("--data_path", type=str, default="/home/v-oostapenko/dev/natural-instructions/tasks")   
# @click.option("--dataset", )
# @click.option("--config_path", type=str, default="/home/v-oostapenko/dev/mttl/configs/llama/finetune_full_lora.json")
@click.option("--model_name", type=str, default="cxwgazou_lora4r") #chavinlo/alpaca-native") yahma/llama-7b-hf chainyo/alpaca-lora-7b togethercomputer/RedPajama-INCITE-Base-7B-v0.1
@click.option("--batch_size", type=int, default=3) 
@click.option("--out_prefix", type=str, default="")      
@click.option("--base", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow") 
@click.option("--from_hf", type=int, default=0)
# @click.option("--nshot", type=int, default=1) # >0 means use canonical examples
@click.option("--model_path", type=str, default="/home/v-oostapenko/dev/amlt/alpaca_poly/alpaca_lora_poly_merge_sep/yahma_llama-7b-hfle2xye5c_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4225.ckpt/loss=0.4225.ckpt") #/home/v-oostapenko/logs/amlt_yahma_llama_atlas_cluster_l1/alpaca4r_topic_ldal1/alpaca-lora_l1/best_model_alpaca_lora_atlas_cluster_te_ada_l1/loss=0.4242.ckpt")#"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hfopq9a3dw_alpaca_lora_full-val/loss=0.5940.ckpt")
@click.option("--example_to_ids_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002.pkl") # depreciated
@click.option("--skill_selector", type=str, default="poly")
@click.option("--nshot", type=int, default=0) # >0 means use canonical examples
@click.option("--reference_file", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/ni/test_references.jsonl") # >0 means use canonical examples
@click.option("--n_tasks", type=int, default=None) # if None, will use a subset of test tasks
@click.option("--usepijma_model_with_llama_adapter", type=int, default=0) # if None, will use a subset of test tasks
def main(model_name="gpt3", batch_size=4, out_prefix="", base="", from_hf=0, model_path="", example_to_ids_path=None, skill_selector="topic", nshot=0, reference_file=None, n_tasks=None, usepijma_model_with_llama_adapter=0):
    task_results = {} 
    topic_router = None 
    data_path = os.getenv("AP_DATA_DIR", "/home/v-oostapenko/dev/natural-instructions/tasks")
    model_dict = { 
        "alpaca_poly_1": {"from_hf":0, "model_name":"alpaca_poly_1", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly1/yahma_llama-7b-hf7btqc8tq_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4240.ckpt/loss=0.4240.ckpt"},
        "alpaca_poly_2": {"from_hf":0, "model_name":"alpaca_poly_2", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly2/yahma_llama-7b-hfz3sxro0n_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4244.ckpt/loss=0.4244.ckpt"},
        "alpaca_poly_3": {"from_hf":0, "model_name":"alpaca_poly_3", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly3/yahma_llama-7b-hfz5pqv3xm_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
        "alpaca_poly_4": {"from_hf":0, "model_name":"alpaca_poly_4", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly4/yahma_llama-7b-hftoqv8su7_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
        "alpaca_poly_5": {"from_hf":0, "model_name":"alpaca_poly_5", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly5/yahma_llama-7b-hflpwsu1yg_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4248.ckpt/loss=0.4248.ckpt"},     
        "alpaca_poly_6": {"from_hf":0, "model_name":"alpaca_poly_6", "depth":0, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_poly_experiment/alpaca_poly6/yahma_llama-7b-hf5y0pj3u9_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4215.ckpt/loss=0.4215.ckpt"},
        "llama": {"from_hf":1, "model_name":"llama", "depth":0, "model_path":""},        
        "cxwgazou_lora4r": {"from_hf":0, "model_name":"cxwgazou_lora4r", "depth":2, "model_path":"/home/v-oostapenko/dev/amlt/alpaca_lora_4r/alpaca_lora_4r/best_model_alpaca_lora_full_r4/loss=0.4216.ckpt"},        
    
    }     
     
    # model_dict = {           
    #     "alpaca_poly_1": {"from_hf":0, "model_name":"alpaca_poly_1", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hf7btqc8tq_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4240.ckpt/loss=0.4240.ckpt"},
    #     "alpaca_poly_2": {"from_hf":0, "model_name":"alpaca_poly_2", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfz3sxro0n_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4244.ckpt/loss=0.4244.ckpt"},
    #     "alpaca_poly_3": {"from_hf":0, "model_name":"alpaca_poly_3", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hfz5pqv3xm_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
    #     "alpaca_poly_4": {"from_hf":0, "model_name":"alpaca_poly_4", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hftoqv8su7_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4205.ckpt/loss=0.4205.ckpt"},
    #     "alpaca_poly_5": {"from_hf":0, "model_name":"alpaca_poly_5", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hflpwsu1yg_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4248.ckpt/loss=0.4248.ckpt"},     
    #     "alpaca_poly_6": {"from_hf":0, "model_name":"alpaca_poly_6", "depth":0, "model_path":"/mnt/amlt_code/models_gcr/yahma_llama-7b-hf5y0pj3u9_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4215.ckpt/loss=0.4215.ckpt"},
    
    # }     
    
    if model_name in model_dict:      
        from_hf = model_dict[model_name]["from_hf"]
        model_path = model_dict[model_name]["model_path"]
        out_prefix = model_name
        
    model_name = "yahma/llama-7b-hf"
    if from_hf==1:             
        if "llama" in model_name:       
            config = {"model": model_name, "model_modifier":None} 
            config = dict_to_dataclass(config)
            model, tokenizer, config = load_model(config, tokenizer_path="yahma/llama-7b-hf") 
            # tokenizer =  LlamaTokenizer.from_pretrained(model_name, padding_side='left')   
            tokenizer =  LlamaTokenizer.from_pretrained("yahma/llama-7b-hf", padding_side='left')   
            tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
            # tokenizer.padding_side='left'       
            model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
            model.model.config.bos_token_id = tokenizer.bos_token_id
            model.model.config.eos_token_id = tokenizer.eos_token_id 
        else:    
            pijama_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)#, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_name)#.to(device)
            tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
            tokenizer.padding_side='left'    
            config = {"model": model_name, "model_modifier":None} 
            config = dict_to_dataclass(config)
            model_class = CLM 
            config.model_object = pijama_model 
            # tokenizer = dm.tokenizer if dm is not None else tokenizer
            model = model_class(**vars(config), tokenizer=tokenizer)      
        
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id   
        model.to(device)
    elif from_hf==0:           
        config = Config()       
        config.model = model_name
        config.n_skills = 1 # by default, can be overwritten in load_model if a checkpoint is provided
        model, tokenizer, config = load_model(config, model_path, device=device)  
        tokenizer.padding_side='left'  
        print(f"Loaded model {model_name} from {model_path}\n")
        print("Loaded config", config.__dict__)
        # if config.example_to_ids_path is not None:    
        #     cluster_result = ClusterResult(config.example_to_ids_path)      
        # if model.args.n_skills>1:          
        #     topic_router = TopicRouter(cluster_with='instruction')
        #     all_topics=topic_router.map.get_topic_data()    
        #     assert cluster_result is not None, "For soft-clustering models, cluster_result must be provided"
        #     assert model.args.n_skills == cluster_result.n_clusters() 
        #     if config.prune_unused_loras:
        #         # prune unused loras
        #         # counts = m = np.bincount(cluster_result._instance.infos.cluster_ids)
        #         skill_ids_to_keep = np.where(
        #             np.bincount(cluster_result._instance.infos.cluster_ids) > 0
        #         )[0]               
        #         model.skill_ids_to_keep = skill_ids_to_keep
        #         # model.model.remove_skills(skill_ids_to_keep)
        #         cluster_result.remove_skills(skill_ids_to_keep) 
        if skill_selector == "average":                  
                topic_router = None
                # skill_ids_to_keep = np.where(np.bincount(cluster_result._instance.infos.cluster_ids)>0)[0]
                # model.model.remove_skills(skill_ids_to_keep) 
                model.model.switch_selector_to_average(selector_to_replace=get_selector(config).__class__)
                model.to(device)      
        
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id        
        model.to(device)          
    elif from_hf==3: #use perf 
        from peft import PeftModel    
        # from inst_follow.models.clm import CLM  
        config = {"model": model_name, "model_modifier":None} 
        config = dict_to_dataclass(config)
        model, tokenizer = load_model(config, device=device) 
        model = PeftModel.from_pretrained(
            model.model,
            "tloen/alpaca-lora-7b",
            device_map={"": device},
        )
        # tokenizer =  LlamaTokenizer.from_pretrained("yahma/llama-7b-hf", padding_side='left')   
        tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
        tokenizer.padding_side='left'
        
        model_class = CLM 
        config.model_object = model 
        # tokenizer = dm.tokenizer if dm is not None else tokenizer
        model = model_class(**vars(config), tokenizer=tokenizer)
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id  
        model.to(device)
    
    if usepijma_model_with_llama_adapter:    
        model.to("cpu")          
        pijama_model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1", torch_dtype=torch.float32)#, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1")#.to(device)
        tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
        tokenizer.padding_side='left'            
        args = copy.deepcopy(config) #{"adapter_modules": "attention", "model_modifier":"llama_adapter"} 
        args.model="togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
        # args = dict_to_dataclass(args) 
        args.adapter_modules = "attention"
        pijama_model = modify_transformer(pijama_model, args)
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
        args.model_object = pijama_model 
        # tokenizer = dm.tokenizer if dm is not None else tokenizer
        model = model_class(**vars(args), tokenizer=tokenizer)
        model.to(device)
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id  
    
    
    # print(config)
    print("Arguments:\n")
    print(data_path, "\n", model_name, "\n", batch_size, "\n", out_prefix, "\n", from_hf, "\n", model_path, "\n", skill_selector)
    # nshot = 1
    
    dataset = load_dataset("yahma/alpaca-cleaned")["train"]
    with open("/home/v-oostapenko/dev/mttl/inst_follow/eval/si_valid_idxs.json") as f:
        idxs = json.load(f)
    dataset = dataset.select(idxs)    
    # shuffle
    # dataset = dataset.shuffle(seed=42)
    rng = np.random.RandomState(42)
    for nshot in [nshot]:#, 5, 10, 15, 20]:      
        out_file_name = f"si_pred_{out_prefix}_{model_name}si-nshot{0}.jsonl" if nshot == 0 else f"si_pred_{out_prefix}_{model_name}si-{nshot}shot.jsonl"
        out_file_name=out_file_name.replace("/", "_")  
        out_file_name = out_file_name.strip()
        base = os.getenv("AMLT_OUTPUT_DIR", "/home/v-oostapenko/dev/mttl/inst_follow")
        
        
        task_results_existing=None
        
        # create directory if doesnt exist
        if not os.path.exists(f"{base}/eval/si"):
            # create
            os.makedirs(f"{base}/eval/si")
        
        
        batch = []
        results = []
        examples = []       
        # shuffle
        dataset = dataset.shuffle(seed=42)
        # select first 300 examples in dataset
        dataset = dataset.select(range(300))
        for i,inst in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            # print(f"{i}/{len(dataset)}")
            context = []         
            if nshot>0: 
                #sampel random examples from dataset
                context_idxs = rng.choice(len(dataset), size=nshot, replace=False)
                context = [dataset[int(idx)] for idx in context_idxs]            
            sample = format(inst, context)
            
            batch.append(sample)
            examples.append(inst)          
            from torchmetrics.text.rouge import ROUGEScore
            if len(batch) == batch_size: 
                gens = generate_outputs(model, batch, tokenizer, temperature=0.7, topic_router=topic_router, skill_selector=skill_selector)
                for pred, example in zip( gens, examples):
                    scorer = ROUGEScore(rouge_keys="rougeL", use_stemmer=True)
                    scores = scorer(pred, example["output"])
                    # scores_cleaned
                    pred2 = pred
                    if "### Instruction" in pred:
                        pred2 = pred.split("### Instruction")[0]
                    scores_cleaned = scorer(pred2, example["output"])
                    inpt = example["instruction"]
                    if len(example["input"])>0:     
                        inpt += f"\n### Input: {example['input']}"
                    results.append({"prediction": pred, 
                                    "rediction_cleaned": pred2,
                                    "ground_truth": example["output"], 
                                    "instruction": example["instruction"],
                                    "input": example["input"],
                                    "output": example["output"], 
                                    "rouge_L": scores["rougeL_fmeasure"].item(), 
                                    "rouge_L_cleaned": scores_cleaned["rougeL_fmeasure"].item()})

                batch = []
                examples = []

        if len(batch):  
            gens = generate_outputs(model, batch, tokenizer, temperature=0.7, topic_router=topic_router, skill_selector=skill_selector)
            for pred, example in zip( gens, examples):
                    scorer = ROUGEScore(rouge_keys="rougeL", use_stemmer=True)
                    scores = scorer(pred, example["output"])
                    # scores_cleaned
                    pred2 = pred
                    if "### Instruction" in pred:
                        pred2 = pred.split("### Instruction")[0]
                    scores_cleaned = scorer(pred2, example["output"])
                    inpt = example["instruction"]
                    if len(example["input"])>0:
                        inpt += f"\n### Input: {example['input']}"
                    results.append({"prediction": pred, 
                                    "rediction_cleaned": pred2,
                                    "ground_truth": example["output"], 
                                    "instruction": example["instruction"],
                                    "input": example["input"],
                                    "output": example["output"], 
                                    "rouge_L": scores["rougeL_fmeasure"].item(), 
                                    "rouge_L_cleaned": scores_cleaned["rougeL_fmeasure"].item()})

            batch = []
            examples = []
        
        #average rouge_L 
        av_rouge_L = np.mean([r["rouge_L"] for r in results])
        av_rouge_L_cleaned = np.mean([r["rouge_L_cleaned"] for r in results])
        print(f"Average rouge_L: {av_rouge_L}\n")
        print(f"Average rouge_L_cleaned: {av_rouge_L_cleaned}")
        results.append({"average_rouge_L": av_rouge_L})
        results.append({"average_rouge_L_cleaned": av_rouge_L_cleaned})
        with open(f"{base}/eval/si/{out_file_name}", "a") as f:
            for r in results:
                f.write(json.dumps(r)+"\n")
            
if __name__ == '__main__':
    main()