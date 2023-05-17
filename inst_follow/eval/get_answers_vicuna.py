# from 
import argparse 
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
import torch
import os
import json 
import click   
from tqdm import tqdm
import sys
import shortuuid
import ray
import numpy as np
import openai
from inst_follow.models.clm import CLM   
from mttl.datamodule.alpaca_data_module import AlpacaDataModule

# sys.path.append("/home/v-oostapenko/dev/FastChat")  
from fastchat.conversation import get_default_conv_template
from fastchat.serve.inference import compute_skip_echo_len
from fastchat.utils import disable_torch_init
from inst_follow.finetune_llama import parse_config, Config
from mttl.models.poly import get_selector
from inst_follow.utils import load_model, TopicRouter
from mttl.cluster_tuning.cluster_reader import ClusterResult

#dd oarams above with click
@click.command()
@click.option("--model_path", type=str, default="/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hfopq9a3dw_alpaca_lora_full-val/loss=0.5940.ckpt")
@click.option("--model_name", type=str, default="yahma/llama-7b-hf")
# @click.option("--eval_on", type=str, default="vacuna", help="vacuna or qna")
@click.option("--model-id", type=str, default="alpaca")
@click.option("--question-file", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/table/vicuna_question.jsonl")
@click.option("--answer-file", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/table/answer/vicuna_80/answers_alpaca7b_lorar4_soft_cluster_atlas_topic.jsonl")#answers_alpaca_lora_full_7b.jsonl")
@click.option("--num-gpus", type=int, default=1)
@click.option("--skill_selector", type=str, default="average", help="none, topic or average")
@click.option("--cluster_with", type=str, default="instruction", help="full, instruction or output")
@click.option("--example_to_ids_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002.pkl")
# @click.option("--embeddings_path", type=str, default="/home/v-oostapenko/dev/mttl/compositional_adapters/data/self_instruct_GPT3")
def run_eval(model_path, model_name, model_id, question_file, answer_file, num_gpus, skill_selector, cluster_with, example_to_ids_path):
    # split question file into num_gpus files
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    
      
    config = Config()       
    config.model = model_name
    config.n_skills = 1 # by default, can be overwritten in load_model if a checkpoint is provided
    model, tokenizer = load_model(config, model_path)  
    print(f"Loaded model {model_name} from {model_path}\n")
    print("Loaded config", config.__dict__)
    if example_to_ids_path is not None:
        cluster_result = ClusterResult(example_to_ids_path) 
    assert cluster_with=='instruction', 'Only instruction clustering is supported for now, otherwise it will overwrite the current atlas project'           
    topic_router = TopicRouter(cluster_with=cluster_with)
    all_topics=topic_router.map.get_topic_data()        
    if model.args.n_skills>1:
        assert cluster_result is not None, "For soft-clustering models, cluster_result must be provided"
        assert model.args.n_skills == len(all_topics) == cluster_result.n_clusters() 
            
        if skill_selector == "average":                  
                topic_router = None   
                skill_ids_to_keep = np.where(np.bincount(cluster_result._instance.infos.cluster_ids)>0)[0]
                model.model.remove_skills(skill_ids_to_keep)
                model.model.switch_selector_to_average(selector_to_replace=get_selector(config).__class__)
                
    tokenizer.padding_side="left"
    # trainer = Trainer(
    #     gpus=-1,   
    #     accelerator="gpu",   
    #     num_sanity_val_steps=5,
    #     amp_backend="native",
    #     default_root_dir=config.output_dir,
    #     max_epochs=config.num_train_epochs,   
    #     max_steps=config.total_steps + 1 if config.total_steps != -1 else -1,
    #     gradient_clip_val=config.max_grad_norm,
    #     log_every_n_steps=20,
    #     strategy=config.compute_strategy if config.compute_strategy else None,
    #     accumulate_grad_batches=config.gradient_accumulation_steps,
    #     precision=int(config.precision)
    #     if config.precision in ["16", "32"]
    #     else config.precision
    # ) 
    # trainer.validate(model, dm)       
    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    ans_jsons = []  
    # if eval_on=="qna":
    #     model, tokenizer = load_model(config, model_path, dm)
    #     for i in range(0, len(ques_jsons), chunk_size):           
    #         ans_jsons.extend(get_model_answers_qna(model,tokenizer))
            
    # elif eval_on=="vacuna":        
    for i in range(0, len(ques_jsons), chunk_size):
        ans_jsons.extend(
            get_model_answers(
                model, tokenizer, topic_router, model_id, ques_jsons[i : i + chunk_size]
            ))

    # ans_jsons = []
    # for ans_handle in ans_handles:
    #     ans_jsons.extend(ray.get(ans_handle))

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")

# @ray.remote(num_gpus=1)
@torch.inference_mode()  
def get_model_answers(model, tokenizer, topic_router, model_id, question_jsons):
    disable_torch_init()
    ans_jsons = []         
    model.cuda()
    batch_size=5
    model.model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.model.config.bos_token_id = 1
    model.model.config.eos_token_id = 2     
    for idxs in tqdm(range(0, len(question_jsons), batch_size)):
        lines = question_jsons[idxs : idxs + batch_size]
        convs = []
        prompts=[]
        questions=[]
        q_indexs=[]
        for line in lines:
            ques_json = json.loads(line)
            idx = ques_json["question_id"]
            qs = ques_json["text"]     
            conv = get_default_conv_template(model_id).copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()             
            questions.append(qs)
            convs.append(conv)
            prompts.append(prompt)    
            q_indexs.append(idx)
        inp = tokenizer(prompts,padding=True, return_tensors="pt")   
        input={
            "input_ids": inp.input_ids.cuda(),
            "task_ids": torch.zeros(batch_size, dtype=torch.long).cuda()*-1,
        }             
        if topic_router:        
            probs = topic_router(questions)   
            input["distances"] = probs#[None,...]
            
        output_ids = model.generate(
            input,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)#[0]
        for i, out in enumerate(outputs):
            conv = convs[i]
            prompt = prompts[i]
            skip_echo_len=compute_skip_echo_len(model_id, conv, prompt)

            out=out[skip_echo_len:].strip()
            ans_id = shortuuid.uuid()
            ans_jsons.append(
                {
                    "question_id": q_indexs[i],
                    "text": out,
                    "answer_id": ans_id,
                    "model_id": model_id,
                    "metadata": {},
                }
            )
    return ans_jsons
    
# @ray.remote(num_gpus=1)
@torch.inference_mode()    
def get_model_answers_qna(model:CLM,tokenizer, *args, **kwargs):
    # model.eval()
    model.cuda()
    ans_jsons = [] 
    instruction = [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]    
    for i, line in enumerate(tqdm(instruction)):
        qs = line               
        qs = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\
            \n### Instruction: {qs}\
            \n### Response:"
        inputs = tokenizer([qs])   
        model.model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.model.config.bos_token_id = 1
        model.model.config.eos_token_id = 2        
        # input_hash = hash_example(qs)
        # instruction_hash = hash_example(qs)
        input={
            "input_ids": torch.as_tensor(inputs.input_ids).cuda(),
            "target_ids": torch.as_tensor(inputs.input_ids).cuda(),
            "task_ids": torch.zeros(1, dtype=torch.long).cuda()*-1,
        }             
        output_ids = model.generate(
            input,  
            # num_beams=4,
            # top_p = 0.75,
            do_sample=False,
            # temperature=0.1,
            # top_k=40,
            max_new_tokens=512)
        outputs = tokenizer.batch_decode(output_ids)[0]
        outputs = outputs.split("### Response:")[1].strip()
        print(outputs)
        # ans_id = shortuuid.uuid()
        # ans_jsons.append({"question_id": idx,
        #                   "text": outputs,
        #                   "answer_id": ans_id,
        #                   "model_id": model_id,
        #                   "metadata": {}})
    # return ans_jsons


# if __name__ == "__main__":       
#     parser = argparse.ArgumentParser(add_help=False)      
#     parser.add_argument("--model-path", type=str, default="/home/v-oostapenko/logs/llama_alpaca/lora_full/best_model.ckpt")
#     parser.add_argument("--model-id", type=str, default="alpaca")  
#     parser.add_argument("--eval_on", type=str, default="vacuna", choices=["vacuna", "qna"]) 
#     parser.add_argument("--question-file", type=str, default="/home/v-oostapenko/dev/mttl/compositional_adapters/eval/vicuna_questions/table/question.jsonl") 
#     parser.add_argument("--answer-file", type=str, default="/home/v-oostapenko/dev/mttl/compositional_adapters/eval/vicuna_questions/table/answer/answers_alpaca_lora_full_7b.jsonl")
#     parser.add_argument("--num-gpus", type=int, default=1)   
#     parser.add_argument("--skill_selector", type=str, choices=["none", "topic", "average"], default="topic")
#     parser.add_argument("--cluster_with", type=str, default="instruction", choices=["full", "instruction", "output"])
#     parser.add_argument("--embeddings_path", type=str, default="/home/v-oostapenko/dev/mttl/compositional_adapters/data/self_instruct_GPT3")
#     config, args = parse_config(parent=parser, return_parser=True)   
#     # args=config     
#     # args = parser.parse_args()
#     ray.init()
run_eval()
