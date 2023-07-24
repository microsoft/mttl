import sys

import json
import pickle
import pandas
import torch
import numpy as np 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sys.path.append("/home/v-oostapenko/dev/mttl")    
from scipy.stats import entropy as calc_entropy
from mttl.cluster_tuning.encodings import ClusterInfos
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from finetune_llama import Config
from mttl.cluster_tuning.cluster_reader import ClusterResult
from inst_follow.utils import load_model, TopicRouter,disable_torch_init
device = "cuda" if torch.cuda.is_available() else "cpu"
# import matplotlib.pyplot as plt
# import seaborn as sns

import json
import os 
import sys 
import numpy as np
import click
import json
import tqdm
import copy
import torch
import datasets
from types import SimpleNamespace
sys.path.append("/home/v-oostapenko/dev/mttl")
from inst_follow.models.clm import CLM  
from transformers import LlamaTokenizer  
from mttl.models.poly import get_selector     
from mttl.models.modify_model import modify_transformer  
from finetune_llama import parse_config, Config
from inst_follow.utils import load_model, TopicRouter,disable_torch_init
from mttl.cluster_tuning.cluster_reader import ClusterResult
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import openai
openai.api_key = "f48b5a4f15dc4e58991738ab066ba465"

# evaluate each 
df = AlpacaDataModule(Config())
dataset=df.get_dataset()
def get_samples(dataset, idxs):
    samples = []
    gt_responses = []
    for i in idxs:
        sample = dataset[i].input_text
        #discard everything after \n### Response:
        sample, response = sample.split("\n### Response:")
        sample+="\n### Response:"
        samples.append(sample)
        gt_responses.append(response)
    return samples, gt_responses


skill_set = ["writing",
             "roleplay", 
             "comprehension", 
             "common-sense reasoning", 
             "coding", 
             "math", 
             "factual knowledge", 
             "conterfactual reasoning", 
             "navigation",
             "planning",
             "social skills",
             "communication",
             "coordination",
             "negotiation",
             "leadership",]


skill_set = (
    "creativity",
    "problem solving",    
    "factuial knowledge",     
    "knowledge extraction",
    "common-sense reasoning",
    "reflection",
    "categorization",
    "planing",
    "chatting",
    "roleplay",
    "memorization",
    "remembering",
    "programing",
    "math", 
    "writing",
    "reading",
    "careful attention",
    "summarization",
    "storytelling",
    "explanation",
    "coherance",
)


keywords = (
    "calculate",
    "count",
    "roleplay",
    "invent",
    "measure",
    "classify",
    "generate",
    "explain",
    "list",
    "plan",
    "identify",
    "reason",
    "select",
    "summarize",
    "write",
)
topics = (
    "animals",
    "colors",
    "food",
    "math",
    "computer science"
)
    
examples = [  
 "\n###Instrution: Write a haiku about a dog. \n###Plan: understand(Instrution) -> reflect(animals,dogs) -> retrieve_information(dogs,heiku) -> plan(heiku) -> write(haiku,dog). \n###Output: Through the door, running \nHair flowing magically \nWelcomes me, my dog.",
 "\n###Instruction: Explain the rules and strategies of playing a game of chess. \n###Plan: understand(Instrution) -> reflect(games,chess) -> retrieve_information(chess,rules,strategies) -> plan(rules,strategies) -> write(rules,strategies). \n###Output: The game of chess is played on a board with 64 squares. Each player starts with sixteen pieces: eight pawns, two knights, two bishops, two rooks, one queen and one king. The goal of the game is to checkmate the opponent's king. This happens when the king is in check and cannot escape.",
 "\n###Instruction: Read the code snippet below and explain what it does. \n###Input: arr = [1,2,3,4,5]\nfor num in arr:\n    print(num*2). \n###Plan: understand(Instrution) -> reflect(code) -> retrieve_information(coding,programming) -> plan(explenation) -> write(explenation). \nOutput: The code snippet above prints the numbers 1, 2, 3, 4, and 5 multiplied by 2.",
 
]


skill_set = (    
        "technical knowledge",
        "revision",
        "simplification",
        "substitution",
        "financial forecasting",
        "punctuation",
        "content integration",
        "idiomatic knowledge",
        "parsing",
        "factuality",
        "decision-making",
        "patience",
        "attention to details",
        "collaboration",
        "presentation",
        "testing and careful attention",
        "logical reasoning",
        "explanatory language",
        "written communication",
        "randomization",
        "market analysis",
        "encouragement",
        "vocabulary knowledge",
        "positive reinforcement",
        "revision and refinement",
        "creative thinking",
        "on-page optimization",
        "comprehension",
        "selection",
        "reading",
        "evaluation",
        "familiarity with idioms",
        "maintenance",
        "analysis",
        "reflective thinking",
        "explanations",
        "counting",
        "json formatting",
        "reviewing",
        "language skills and grammar",
        "self-assessment",
        # "language understanding",
        "reading and comprehension",
        "summarization",
        "editing",
        "organizing",
        "knowledge of haiku",
        "grammatical knowledge",
        "familiarity with sql",
        "sentence reformation",
        # "local seo",
        # "language and grammar knowledge",
        "understanding number systems",
        "factual knowledge",
        "sql knowledge",
        "synonym substitution",
        "tokenization",
        "roleplay",
        "interpretation",
        # "active listening",
        # "sentence structure knowledge",
        # "factuai knowledge",
        "syntax and grammar",
        "perseverance",
        # "keyword research",
        "pattern recognition",
        "word choice analysis",
        "understanding the context",
        "execution",
        "interconnected thinking",
        "professionalism",
        "mathematical reasoning",
        "syntax transformation",
        "time management",
        "networking",
        "review and revision",
        "grammar and syntax knowledge",
        "remembering",
        "programming",
        "storytelling",
        "transformation",
        "marketing and advertising",
        # "knowledge of present continuous verbs",
        # "fact-based knowledge",
        "decision making",
        "spelling knowledge",
        # "grammar and language understanding",
        "planning",
        "reflection",
        "motivation",
        "rhyme",
        "memory",
        "paraphrasing",
        "common sense reasoning",
        "summarization and paraphrasing",
        "math",
        "understanding",
        "mathematical knowledge",
        # "writing and coherance",
        "memorization",
        "coherence",
        "language translation",
        "organization",
        "updating",
        "imagery",
        # "knowledge of grammar",
        "proofreading",
        "testing the regex pattern",
        "synonyms and vocabulary knowledge",
        "concatenation",
        # "grammar manipulation",
        # "context understanding",
        "language and grammar",
        "problem solving",
        # "factual knowledge (real world)",
        "prioritization",
        "linguistic and grammatical knowledge",
        "quality content",
        "empathy",
        "contextual understanding",
        "language-related knowledge",
        "problem-solving",
        # "understanding the task",
        "planing",
        "editing and proofreading",
        "testing and refining",
        # "analytical and problem-solving skills",
        "creative writing",
        "coherane",
        "vocabulary",
        # "factuil knowledge",
        "empathy and roleplay",
        "attention",
        "rhyme and rhythm",
        "comparing and contrasting",
        # "factual knowledge (star wars)",
        "adaptability",
        "backlinks",
        "analytical reasoning",
        "writing",
        # "mathematical notation knowledge",
        "organizational skills",
        "adaptation",
        "common-sense reasoning",
        "language proficiency",
        # "constructing the regex pattern",
        "creativity",
        "inference",
        "language knowledge",
        "argumentation",
        "empathy and listening",
        "music theory",
        "coherance",
        "grammar and sentence construction",
        "knowledgeof alphabets",
        "descriptive language",
        "translation",
        "chatting",
        "coherent writing",
        "testing",
        "data visualization",
        "social media management",
        "understanding of email structure",
        "grammar knowledge",
        # "factuial knowledge",
        "persuasion",
        "measure and analyze",
        "grammar",
        # "responsive design",
        # "natural language processing",
        "careful attention",
        "fact checking",
        "communication",
        "attention to detail",
        "research",
        "critical thinking",
        "categorization",
        "comparison",
        # "research/reading",
        # "comparison and categorization",
        "comparative analysis",
        "facts contextualization",
        "knowledge extraction",
        "explanation",
        "analytical thinking",
        # "knowledge of regular expressions",
        "programing",
        # "summary and presentation",
        "factoidal knowledge",
        "grammar skills",
        "visualization",
        # "fact-checking",
        "language identification",
        "explanatory",
        "recall",
        "chronological ordering",
        "game-rule knowledge")

skill_set=set(skill_set) 
# def construct_gpt_template(example:dict):  
#     prompt = f"I am trying to cluster instructions/questions based on which skill is needed to answer them. \
#                 \n To do this, given an instruction, please provide a comprehensive set of skills needed to answer it in a situation without any access to extrnal knowledge. Here is an incomplete set of possible skills: {[skill for skill in skill_set]}. \
#                 \n Only use skills that are in the set of skills above.\
#                 \n For each skill you list, please provide a score from 0 to 1 indicating how important that skill is to answer the instruction, the scores of all skills together must sum to 1.\
#                 \n First, think step by step, in order of executions teps, explain why each of the skills is necessary to answer the instruction/question provided below. \
#                 \n At the end of your output provide a comprehensive set of skills as a JSON-formatted dictionary, where the keys are the skills and values are the scores. Order the skills in order of execution. Here is the instruction:\
#                 \n### Instruction: {example['instruction']}."
#     if example['input']:
#         prompt+=f" ### Input: {example['input']}."  
#     prompt+=f"\n### Ground teruth Response: {example['output']}."
#     prompt+="\n### Skills:"
#     return prompt

def construct_gpt_template(examples:dict):                   
    prompt = f"You are given an instruction and the ground truth response in the format ###Instruction:[instruction] (optionally followed by ### Input: [input]), ### Ground teruth Response: [response].\
                \n Given this instruction, your task is to provide a set of keywords that are useful to cluster the instruction. Generate two types of keywords: [skills:] describing the general skills needed to execute the instruction, [topic:] and keywords describning the topics of the instructions.\
                \n For skills, trey using the keywords only from this set but expand it if needed: {[mechanism for mechanism in keywords]}. For topics, try using the keywords from this set but expand it if needed: {[topic for topic in topics]}.\
                \n Remember, both skills and topics must follow from the instruction. Both must be as precise as possible but also general, as they will be used to cluster a dataset of instructions.\
                \n For each keyword, provide a score from 0 to 1 indicating how important that keyword for the instruction, the scores for skills must sum to 1, the scores for topics must also sum to 1.\
                \n First, think step by step, explaining why each of the skills and topics is meaningful. MNake sure you don not unneccessarily create new skills or topics.\
                \n At the end of your reasoning, provide a comprehensive set of keywords as a JSON-formatted dictionary in the format: ### JSON dict skills: [JSON dictionary with skills], ### JSON dict topics: [JSON dictionary with topics], where the \"keys\" are the keywords and values are the scores. Do not include keywords with score 0.\
                \n Here is the instruction: \n"
    for i,example in enumerate(examples):   
        prompt+=f"\n### Instruction: {example['instruction']}."
        if example['input']:
            prompt+=f" ### Input: {example['input']}."  
        prompt+=f"\n### Ground teruth Response: {example['output']}."
    return prompt

def construct_gpt_template(examples:dict):                  
    prompt = f"You are given {len(examples)} instructions from a larger dataset of instructions. For each you are also given the ground truth response. This input is given in the following format: ###Instruction [instruction id]: [insrtruction] (optionally followed by ### Input [instruction id]: [input]), ### Ground teruth Response [instruction id]: [response].\
            \n Your task is to provide a set of keywords that are useful to cluster these instructions. Generate two types of keywords: action -- describing the main action required by an instruction, topic -- keywords describning the topics of the instructions.\
            \n For actions, try using the keywords from this set but expand it if needed: {[mechanism for mechanism in keywords]}. For topics, try using the keywords from this set but expand it if needed: {[topic for topic in topics]}. If generate new general topics if needed, i.e. if topics in the set are not sufficient for the instruction. Also, generate new actions if needed.\
            \n Remember, both actions and topics must follow from the instructions. Both must be as precise as possible but also general, as they will be used to cluster the large dataset of instructions, from which these instructions are sempled.\
            \n For each keyword, provide a score from 0 to 1 indicating how important that keyword is for the instruction, the scores for actions must sum to 1, the scores for topics must also sum to 1.\
            \n In your output, for each instruction first repeat the instruction. Then for the given instruction think step by step, explaining why the actions and topics you list are meaningful. Remember, make sure the topics are correct but are not overly specialized: e.g. do not use the city name as the topic as it would be overly specialized.\
            \n Then provide a comprehensive set of keywords as a JSON-formatted dictionary in the format: ### JSON dict actions [instruction id]: [JSON dictionary with actions], ### JSON dict topics [instruction id]: [JSON dictionary with topics], where the \"keys\" are the keywords and values are the scores. Do not include keywords with score 0.\
            \n Here are the instructions: \n"
    for i,example in enumerate(examples):   
        prompt+=f"\n### Instruction {i}: {example['instruction']}."
        if example['input']:
            prompt+=f" ### Input  {i}: {example['input']}."  
        prompt+=f"\n### Ground teruth Response  {i}: {example['output']}."
    return prompt

openai.api_type = "azure"
openai.api_base = "https://gcrgpt4aoai7.openai.azure.com/"
openai.api_version = "2023-05-15"

new_dataset = {
    "instruction": [],
    "input": [],
    "output": [],
    "skill_set": [],
}
import time
import asyncio         
from inst_follow.operator import GPT  
operator = GPT(model_name="gpt-4")   
requests = []
instructions = []
bs_per_request = 3
batch=[]
b_inst=[]
for example in dataset.dataset:
    batch.append(example)
    b_inst.append(example) 
    if len(batch)==bs_per_request:
        sample = construct_gpt_template(batch)
        requests.append(sample)
        instructions.append(b_inst)
        batch = []
        b_inst = []
if len(batch)>0:   
    sample = construct_gpt_template(batch)
    requests.append(sample)
    instructions.append(b_inst)
    batch = []
    b_inst=[]
    
def extract_json(message):
    json_start = message.find("{")
    json_end = message.find("}")   
    json_string = message[json_start:json_end+1]
    json_string = json_string.lower().replace("_", " ")
    #
    #parse JSON
    skills_dict = json.loads(json_string) 
    skills_used = [skill.strip().lower().replace("_", " ") for skill in skills_dict.keys() if skills_dict[skill]>0]
    # append new skills to skill set
    # skill_set = set(list(skill_set)+skills_used)
    return skills_used, skills_dict
n_seen_inst=0    
def write_to_file(outputs, instructions, start_=0):
    # new_dataset=[]   
    global n_seen_inst             
    for i, (output, inst) in enumerate(zip(outputs, instructions)):  
        # try:                             
        # in each output there are jsons for bs_per_request examples
        global bs_per_request  
        batch_size = output.count("JSON dict actions")
        for j in range(len(inst)):  
            message_topics = output.split(f"dict topics {j}:")[1]
            message_skills = output.split(f"dict actions {j}:")[1]  
            skills_used, skill_dict_dist = extract_json(message_skills)
            topics_used, topics_dict_dist = extract_json(message_topics)
            # append new skills to skill set
            global keywords   
            global topics
            sample = {'instruction': inst[j]["instruction"],
                      'input': inst[j]["input"],
                      'output': inst[j]["output"],
                      'gpt_output': output,
                      'skill_set': skill_dict_dist,
                      'topic_set': topics_dict_dist
                    #   'global_skill_set': list(skill_set)
                    }
            # new_dataset.append(sample)           
            keywords = set(list(keywords)+skills_used)
            topics = set(list(topics)+topics_used)
            # append new entried to dataset json
            with open("new_dataset_3plan.jsonl", "a") as f:
                json.dump(sample, f, indent=4)
        n_seen_inst+=batch_size
        
all_outputs = []
#empty the dataset file
# with open("new_dataset.jsonl", "w") as f:
#     js
#proceed in batches
batch_size = 10     
start_=0             
for end_ in tqdm.tqdm(range(0, len(requests), batch_size)):
    request_batch = requests[start_:end_]
    print(start_, end_, len(request_batch))    
    inst_batch = instructions[start_:end_]      
    outputs = asyncio.run(operator.gather_chat_response(request_batch))
    # outputs = await task
    try:   
    # all_outputs.extend(outputs) 
        write_to_file(outputs,inst_batch,start_)
    #print exception
    except Exception as e:
        print(e)
            
    # print("Skill set", skill_set)
    start_ = end_      
    

# for i,example in tqdm.tqdm(enumerate(dataset.dataset), total=len(dataset.dataset)):
#     # if i==10:
#     #     break
#     template_to_fill = construct_gpt_template(example)
#     print(f"PROMPT:\n{template_to_fill}")
#     failure = True
#     num_retries = 0        
#     while failure and num_retries < 3:
#         cache_row = None
#         try:
#             start = time.perf_counter()  
#             response = openai.ChatCompletion.create(
#                 # model="gpt-3.5-turbo",                
#                 deployment_id="gpt-4",      
#                 # deployment_id="gpt-35-turbo",
#                 # model_name="gpt-4",
#                 messages=[
#                     {"role": "user", "content": template_to_fill},
#                 ],
#             )
#             message = json.loads(str(response.choices[0]))["message"]["content"]
#             print(message)
#             try:
#                 # extract JSON from message
#                 json_start = message.find("{")
#                 json_end = message.find("}")   
#                 json_string = message[json_start:json_end+1]
#                 #parse JSON
#                 skills_dict = json.loads(json_string) 
#                 skills_used = [skill.strip().lower().replace("_", " ") for skill in skills_dict.keys() if skills_dict[skill]>0]
#                 # append new skills to skill set
#                 skill_set = set(list(skill_set)+skills_used)
                
#                 num_retries += 1  
#                 new_dataset["instruction"].append(example["instruction"])
#                 new_dataset["input"].append(example["input"])
#                 new_dataset["output"].append(example["output"])
#                 new_dataset["skill_set"].append(skills_dict)
#                 failure = False
                
#                 if i%10==0:
#                     # save new dataset
#                     with open("new_dataset.json", "w") as f:
#                         json.dump(new_dataset, f)
#             except:                
#                 num_retries += 1
                
#             end = time.perf_counter()
#             if end - start < 1:
#                 time.sleep(1 - (end - start))
#         except:
#             time.sleep(3)
    
# save new dataset
# with open("new_dataset.json", "w") as f:
#     json.dump(new_dataset, f)