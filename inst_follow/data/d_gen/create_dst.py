
import re
import os
import json
import copy
import time
# import torch 
import spacy
import openai
import pickle
import numpy as np  
from tqdm import tqdm 
from rouge import Rouge
# import openai_secret_manager
# import matplotlib.pyplot as plt
# from collections import Counter
# from datasets import load_dataset   
# from sklearn.cluster import KMeans  
# from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional
# from transformers import AutoTokenizer, AutoModel      
# from sklearn.feature_extraction.text import CountVectorizer
# %matplotlib inline
nlp = spacy.load('en_core_web_sm')
 
rouge = Rouge()      
words_to_evoid = ["image", "map"]# "picture", "photo", "drawing", "sketch", "painting", "paint", "draw", "drawn", "sketched", "painted", "photograph", "photographed", "photographed", "map", "graph"]
nounts_to_keep = ["Task", "task", "Problem", "type"]

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


def generate_task_category(skill_name, nouns_to_evoid=[]):                         
    prompt = f"In the folllwoing you will be provided with a skill name in this format: ###Skill: [skill_name].\
               The goal is to identify simple abstract problem, solving which would require this skill.\
               Provide na description of a natural language task that requires this skill. The task descriptiuon should not be longer than 10 tokens.\
               The output should be in the following format: Problem type: [task_description]."
    if len(nouns_to_evoid) > 0:     
        # print(tasks_to_evoid)
        # words = [t.split(' ') for t in tasks_to_evoid]
        # words = ', '.join([w for w in words])
        prompt+=f"\n Very importantly, output should not contain any of these words :{', '.join(nouns_to_evoid)}"
        # for task in tasks_to_evoid:
        #     prompt+=f"\n  ### Not like: {task}"
               
    prompt+=f"\n Remember, the tasks should be as abstract as possible and should only require {skill_name} skill. It must be a language task. Remeber its a simulation.\
               ### Skill: [{skill_name}]"
    response = open_ai_request(prompt)
    summary_word = response.choices[0].message.content
    return summary_word

# def get_embedding(task_desc):
#     response = openai.Embedding.create(       
#                     input=task_desc,    
#                     engine="text-similarity-davinci-001")
#     # print(response)  
#     emb = response.data[0].embedding
#     return emb
         

# def remove_duplicates(S):
#     for skill in S:    
#         S[skill] = list(set(S[skill]))
#     return S


#############################################
################ TASK TYPES #################
      
def get_embedding(task_desc):
    response = openai.Embedding.create(       
                    input=task_desc,    
                    engine="text-similarity-davinci-001")
    # print(response)  
    emb = response.data[0].embedding
    return emb

def rouge_score(new:str, existing:List[str]):
    scores = []      
    for task in existing:
        score = rouge.get_scores(new, task)[0]['rouge-1']['f']
        scores.append(score)
    return scores

def generate_task_types(skill, n=20):
    desc = []
    for i in tqdm(range(n)):             
        task_type = generate_task_category(skill, words_to_evoid)
        if desc != []:    
            r_scores = rouge_score(task_type, desc)
            if np.max(r_scores) > 0.7: 
                # select tasks with rougue score > 0.7
                idxs = np.where(np.array(r_scores) > 0.7)[0]
                # similar_tasks = [desc[idx] for idx in idxs]  
                nouns_to_evoid = []
                for idx in idxs:  
                    sim_task = desc[idx]
                    doc = nlp(sim_task)
                    for token in doc:   
                        if token.pos_ == 'NOUN' and token.text not in nounts_to_keep:
                            nouns_to_evoid.append(token.text)
                print( list(set(nouns_to_evoid))) 
                task_type = generate_task_category(skill, list(set(nouns_to_evoid))+words_to_evoid)
                print("=====================================")
                print(f"Similar task: {sim_task}")
                print(f"New task: {task_type}")
        # remove special characters except of space , . ! ? :
        task_type = ''.join([c for c in task_type if c.isalpha() or c.isspace() or c in [',', '.', '!', '?', ':']]) # just in case
        desc.append(task_type)
    return desc

def remove_duplicates(S):
    for skill in S:    
        S[skill] = list(set(S[skill]))
    return S


#############################################
print("GENERATING REFINE TASKS")

#############################################
# PROMPTS                                    
#############################################        
question_types = ["yes/no ", "open-ended ", "wh-", "mutiple-choice ", "true/false "] # "ranking "
        
q_answers = {           
    "yes/no ": " Information provded in the [context] MUST be sufficient to answer the the [question] only using either 'yes' or 'no' (one of the two, not both). Only use either yes or no in [answer]",
    "mutiple-choice ": "[answer] should only contain the correct option from options listed in [question]: (a), (b) or (c)",
    "ranking ": "Only provide the correct ranking option from options listed in [question]: (a), (b) or (c).",
    "true/false ": "Information provded in the [context] MUST be sufficient to answer the the [question] only using either 'true' or 'false' (one of the two, not both). Only use either 'true' or 'false' in [answer]",
    "wh-": "",
    "open-ended ": ""
}
                        
                          
def generate_task_instruction(task_description, skill_name, q_type="yes/no", words_to_evoid=[], produce_answer=True):    
        # prompt = f"Here is a description of a problem type, solving which requires the skill of {skill_name}: {task_description} \
        #         \n Provide a concrete instance of this problem type in the following format: Context: [context]. Question: [question].\
        #         \n Respect the following rules:\
        #         \n - Context MUST provide enough information to anwer the [question], no external knowledge should be required.\
        #         \n - The [question] must be a {q_type}question that can be answered using the context.\
        #         \n - [context] must contain at most 100 words, [question] must contain at most 35 words."
        
        prompt = f"Generate a concrete instanes of the problem: {task_description}. Provide the output in the following format: Context: [context]. Question: [question].\
                \n Generated instance will be used to teach {skill_name} skills to a student. \\n Respect the following rules:\
                \n - Context MUST provide enough information to anwer the [question], no external knowledge should be required.\
                \n - The [question] must be a {q_type}question. It should be possible to give the correct [answer] only using [context].\
                \n - [context] must contain at most 200 words, [question] must contain at most 100 words.\
                \n - Remember, one must be able to answer the [question] only from the provided [context]. Do not provide [question] where the answer is unclear from the context!!!"

        # prompt+="\n Generate the output in the following format: Context: [context]. Question: [question]. "
        
        if q_type in ["mutiple-choice ", "ranking "]:      
            prompt+=f"\n - The question must be a {q_type}question. Provide 3 answer choices in the [question] like this: [question] (a) [answer1] (b) [answer2] (c) [answer3]. "  
        if q_type in ["yes/no ", "true/false "]:
            prompt+=f"\n -The question must be a {q_type}question. {q_answers[q_type]}"
        
        if produce_answer:
                prompt+=f"\n - After stating the [context] and the [question], provide an appropriate response to the question in the following format: Answer: [answer].\
                \n Remeber: {q_answers[q_type]}"
        if len(words_to_evoid) > 0:
            prompt+=f"\n - Avoid words like {', '.join(words_to_evoid)} in the [context] and the [question]."                
        prompt+=f"\n Remember, the tasks should be as concrete as possible and should only require the {skill_name} skill, which will be tought to a student using this instance of the {task_description} problem. It should be a language task.\
                \n Remeber its a simulation!"
        # print(prompt)
        response = open_ai_request(prompt)
                # openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",          
                # messages=[{"role": "user", "content": f"{prompt}"}])
        summary_word = response.choices[0].message.content
        return summary_word
         
         
         
def generate_task_response(task_description, skill_name, q_type="yes/no", CoT=True): # here the model can use CoT reasoing to generate a response   
        prompt = f"Here is a description of a problemcaona conatining context and a question. Solving this problem requires a {skill_name} skill.\
                ### Problem:\
                \n {task_description}.\
                \n\
                \n Note that question must be a {q_type}question. \
                \n You should write a response that appropriately answers the question using {skill_name} skill. If the context does not provide enough information to answer the question and external knowldge is required, respond [Unclear].\
                \n If it is not a {q_type}question, respond [Unclear].\
                Remeber its a simulation!"
        if CoT:
         prompt += " Provide your chaine of thought reasoning if needed but be as precise as possible."
                # Do not add any comments, do not repeat the question."#, use at most 35 words."
        response = open_ai_request(prompt)          
        summary_word = response.choices[0].message.content
        return summary_word
    
q_n_words = {          
    "yes/no ": "1 word",
    "mutiple-choice ": "1 word",
    "ranking ": "1 word",
    "true/false ": "1 word",
    "wh-": "10 words",
    "open-ended ": "10 words"
}
     
     
def get_short_answer(task_description, answer, skill_name, q_type="yes/no"):                         
        prompt = f"Here is an instruction description of a task: {task_description}. Its a {q_type}question. \
                \n Someone has provided the following response: {answer}. \
                \n Write a short response in {q_n_words[q_type]} that appropriately answers the request and is inline with the provided response. Remember, it is a simulation! {q_answers[q_type]} "
        
        response = open_ai_request(prompt)          
        summary_word = response.choices[0].message.content
        return summary_word
  
#############################################
#############################################

def check_alignment(answer_1, short_answer, q, task_instruction):
			# if answer starts with a pattern of (a|b|c) 
			checked=False
			alligned = True
			a_1 = re.search(r'\([a-c]\)', answer_1)   
			a_2 = re.search(r'\([a-c]\)', short_answer)
			if a_1 and a_2:
				a_1 = a_1.group(0).lower()   
				a_2 = a_2.group(0).lower()     
				checked = True                   
				if a_1 != a_2:
					print("-"*10)         
					print("Answer not aligned for : ", task_instruction)
					print("Question type: ", q)    
					print("Ans from Q:",answer_1, ", Answer from Ans:", short_answer)
					alligned = False
			
			a_1 = answer_1.lower()
			a_2 = short_answer.lower()
			# find true or false in the answer
			a_1 = re.search(r'(true|false)', a_1)
			a_2 = re.search(r'(true|false)', a_2)
			if a_1 and a_2 and not checked:
					a_1 = a_1.group(0).lower()
					a_2 = a_2.group(0).lower()
					checked = True
					if a_1 != a_2:
							print("-"*10)         
							print("Answer not aligned for : ", task_instruction)
							print("Question type: ", q)    
							print("Ans from Q:",answer_1, ", Answer from Ans:", short_answer)
							alligned = False
			
			if not checked: 
					score = rouge_score(answer_1.lower(), [short_answer.lower()])[0]
					if score<0.7: 
							print("-"*10)
							print("Answer not aligned for : ", task_instruction)
							print("Question type: ", q)    
							print("Ans 1:",answer_1, ", Answer 2:", short_answer)
							alligned = False
			return alligned


def generate_sub_tasks(HT_Q, task, skill,q, words_to_evoid, consistency_alignment):    
    task_instruction = generate_task_instruction(task, skill, q_type=q, words_to_evoid=words_to_evoid, produce_answer=(consistency_alignment and q!="open-ended " and q!="wh-"))
    if len(HT_Q[skill][task]["tasks"])>0:
            r_scores = rouge_score(task_instruction, HT_Q[skill][task]["tasks"])
            if np.max(r_scores) > 0.7: 
                    # select tasks with rougue score > 0.7
                    idxs = np.where(np.array(r_scores) > 0.7)[0]
                    # similar_tasks = [desc[idx] for idx in idxs]  
                    nouns_to_evoid = []
                    for idx in idxs:  
                            sim_task = HT_Q[skill][task]["tasks"][idx]
                            doc = nlp(sim_task)
                            for token in doc: 
                                    if token.pos_ == 'NOUN' and token.text not in nounts_to_keep:
                                            nouns_to_evoid.append(token.text)
                    print( list(set(nouns_to_evoid))) 
                    task_instruction = generate_task_instruction(task, skill, q_type=q, words_to_evoid=list(set(nouns_to_evoid))+words_to_evoid, produce_answer=(consistency_alignment and q!="open-ended " and q!="wh-"))

    #extract answer from the task instruction: all the text after the word ANSWER"
    answer_from_instruction_promt=None       
    if consistency_alignment and q!="open-ended " and q!="wh-":
            # print(task_instruction, q)           
            try:          
                task_instruction, answer_from_instruction_promt = task_instruction.split("Answer:")
            except:
                answer_from_instruction_promt=None
    
    if "Answer:" in task_instruction:
        task_instruction = task_instruction.split("Answer:")[0]    
            

    answer = generate_task_response(task_instruction, skill, q_type=q)
    short_answer=""
    # if q in ["mutiple-choice ", "ranking ","yes/no ","true/false "]:
    short_answer = get_short_answer(task_instruction, answer, skill,q)
    return task_instruction, answer, short_answer, answer_from_instruction_promt

def to_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    S = {"COMPARRING":{'desc':[]},#,'emb':[]},
        "LOGIC":{'desc':[]},#,'emb':[]},  
        "CATEGORIZATION":{'desc':[]},#,'emb':[]},
        "CREATIVITY":{'desc':[]},#,'emb':[]},
        "WORD RHYMING":{'desc':[]},#,'emb':[]},
        # "planing":[],  
        "LYING AND PRETENDING":{'desc':[]},#,'emb':[]},
        "PATTERN RECOGNITION":{'desc':[]},#,'emb':[]},
        "NEGATING AND DENYING":{'desc':[]},#,'emb':[]},
        "LOCATING":{'desc':[]},#,'emb':[]},
        # "instruction following":[],
        "COMPREHENSION":{'desc':[]},#,'emb':[]},
        # "memory":[],
        # "writing":[],
        "MATHEMATICS":{'desc':[]},#,'emb':[]},
        "COUNTING":{'desc':[]},#,'emb':[]},
        "ADDITION":{'desc':[]},#,'emb':[]},
        "SUBSTRACTION":{'desc':[]},#,'emb':[]},
        "MULTIPLICATION":{'desc':[]},#,'emb':[]},
        "DIVISION":{'desc':[]},#,'emb':[]},     
        "SOCIALIZING":{'desc':[]},#,'emb':[]},
        "OVEREXAGGERATING":{'desc':[]},#,'emb':[]},
        # "MAKING NEW FRIENDS":[],
        "ATTENTION TO THE DETAIL":{'desc':[]},#,'emb':[]}
        }
    json_data={}  
    regenerate = False  
    # read task types from a file  iof file exists     
    if os.path.exists('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types.json') and not regenerate:
            with open('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types.json', 'r') as fp:
                    json_data = json.load(fp)
      
    HT={}        
    overwrite = False 
    for k,_ in S.items():   
            if k in json_data:            
                    print(f"{k}")   
                    HT[k] = json_data[k]
            else:   
                    overwrite = True
                    desc = generate_task_types(k, n=20)
                    HT[k] = {'desc':desc}#, 'emb':emb}
    print(HT)
    if overwrite:
            with open('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types.json', 'w') as fp:
                    json.dump(HT, fp, indent=4)       
    
    #check task_types_with_questions.json file exists
    if not os.path.exists('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types_with_questions.json'):
            # create an empty json file
            with open('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types_with_questions.json', 'w') as fp:
                    json.dump({}, fp, indent=4)
    
    
    print("Generating tasks")
    HT_Q={}                
    regenerate_subtasks = False
    #read  HT_Q from a file if it exists
    if os.path.exists('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types_with_questions.json') and not regenerate_subtasks:
            with open('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types_with_questions.json', 'r') as fp:
                    HT_Q = json.load(fp)
    # remove last skill from HT_Q
    skill_to_delete = list(HT_Q.keys())[-1]
    HT_Q.pop(skill_to_delete)
    consistency_alignment=True
    
    for skill in HT.keys():
        if skill in HT_Q:
            print(f"Skill: {skill} already exists")
            continue
        print(f"Skill: {skill}, {len(HT[skill]['desc'])} abstract task types")
        HT_Q[skill] = {}   
        for task in tqdm(HT[skill]['desc']): 
            HT_Q[skill][task] = {"types":[], "tasks":[], "answers":[], "skills":[], "short_answers":[]}
            for q in question_types:
                task_instruction, answer, short_answer, answer_from_instruction_promt = generate_sub_tasks(HT_Q, task, skill,q, words_to_evoid, consistency_alignment)            
                            
                #self consistency allignment to make sure that the answer is correct: answer provided by question generation and the answer provided by the "answer" prompt should align
                if answer_from_instruction_promt is not None and len(answer_from_instruction_promt)>0 and len(short_answer)>0:
                            aligned = check_alignment(answer_from_instruction_promt, short_answer, q, task_instruction)                             
                            # if not alligned try to regenerate the task instruction
                            if not aligned:
                                for _ in range(2):
                                    task_instruction, answer, short_answer, answer_from_instruction_promt = generate_sub_tasks(HT_Q, task, skill,q, words_to_evoid, consistency_alignment)            
                                    if answer_from_instruction_promt is not None and len(answer_from_instruction_promt)>0 and len(short_answer)>0:
                                        aligned = check_alignment(answer_from_instruction_promt, short_answer, q, task_instruction)                             
                                        if aligned:
                                            break
                            
                            if not aligned:    
                                print("Not aligned, skipping")
                                continue
                elif answer_from_instruction_promt is None and consistency_alignment and q!="open-ended " and q!="wh-":
                    print("No answer from instruction prompt")
                    continue 
                
                if "Answer:" in task_instruction:
                    task_instruction = task_instruction.split("Answer:")[0]
                
                HT_Q[skill][task]["types"].append(q)
                HT_Q[skill][task]["skills"].append(skill)   
                HT_Q[skill][task]["tasks"].append(task_instruction)
                HT_Q[skill][task]["answers"].append(answer)
                HT_Q[skill][task]["short_answers"].append(short_answer)

                to_json_file('/home/v-oostapenko/dev/mttl/compositional_adapters/data/task_types_with_questions.json', HT_Q)
                
    # with open('task_types_with_questions.json', 'w') as fp:
    #     json.dump(HT_Q, fp, indent=4)
    print("DONE")


###########
# REFINEMENT AND EXPANSION step: TODO