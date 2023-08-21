import json
import os 
import pickle
import numpy as np
import click
import json
import tqdm
import copy
import torch  
import sys    
import glob        
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..",".."))
from projects.instr_routing.eval.model_dict import model_dict

print(sys.path)
print("getdefaultencoding: ",sys.getdefaultencoding())

# print the output of system command
# print(os.popen('ls -l /mnt/').read())
# print(os.popen('ls -l /').read())
# print(os.popen('ls -l /home').read())
# print(os.popen('ls -l /mnt/default/data').read())    
# print(os.popen('ls -l /mnt/default/data/natural-instructions/tasks').read())
   
from projects.instr_routing.models.clm import CLM
from mttl.dataloader import ni_metrics   
from transformers import LlamaTokenizer  
from mttl.models.poly import get_selector           
from mttl.models.modify_model import modify_transformer  
from projects.instr_routing.finetune_llama import parse_config, Config      
from projects.instr_routing.utils.utils import load_model, disable_torch_init
from projects.instr_routing.cluster_tuning.cluster_reader import ClusterResult

from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
def dict_to_dataclass(d):
    from dataclasses import make_dataclass
    return make_dataclass("X", d.keys())(**d)
  
test_tasks = [
        "task1356_xlsum_title_generation",
        "task893_gap_fill_the_blank_coreference_resolution",
        "task641_esnli_classification",
        "task1529_scitail1.1_classification",
        "task202_mnli_contradiction_classification",
        "task670_ambigqa_question_generation",
        "task1393_superglue_copa_text_completion",
        "task1344_glue_entailment_classification",
        "task288_gigaword_summarization",
        "task1387_anli_r3_entailment",
        "task1664_winobias_text_generation",
        "task1161_coda19_title_generation",
        "task880_schema_guided_dstc8_classification",
        "task738_perspectrum_classification",
        "task1439_doqa_cooking_isanswerable",
        "task645_summarization",
        "task619_ohsumed_abstract_title_generation",
        "task1728_web_nlg_data_to_text",
        "task1640_aqa1.0_answerable_unanswerable_question_classification",
        "task648_answer_generation",
        "task242_tweetqa_classification",
        "task620_ohsumed_medical_subject_headings_answer_generation",
        "task1159_bard_analogical_reasoning_containers",
        "task500_scruples_anecdotes_title_generation",
        "task890_gcwd_classification",
        "task039_qasc_find_overlapping_words",
        "task1154_bard_analogical_reasoning_travel",
        "task1612_sick_label_classification",
        "task1442_doqa_movies_isanswerable",
        "task233_iirc_link_exists_classification",
        "task936_defeasible_nli_snli_classification",
        "task1386_anli_r2_entailment",
        "task1152_bard_analogical_reasoning_causation",
        "task290_tellmewhy_question_answerability",
        "task304_numeric_fused_head_resolution",
        "task760_msr_sqa_long_text_generation",
        "task035_winogrande_question_modification_person",
        "task569_recipe_nlg_text_generation",
        "task391_causal_relationship",
        "task891_gap_coreference_resolution",
        "task1586_scifact_title_generation",
        "task602_wikitext-103_answer_generation",
        "task1195_disflqa_disfluent_to_fluent_conversion",
        "task1409_dart_text_generation",
        "task033_winogrande_answer_generation",
        "task1407_dart_question_generation",
        "task402_grailqa_paraphrase_generation",
        "task201_mnli_neutral_classification",
        "task520_aquamuse_answer_given_in_passage",
        "task892_gap_reverse_coreference_resolution",
        "task828_copa_commonsense_cause_effect",
        "task769_qed_summarization",
        "task1155_bard_analogical_reasoning_trash_or_treasure",
        "task1385_anli_r1_entailment",
        "task1531_daily_dialog_type_classification",
        "task1516_imppres_naturallanguageinference",
        "task1394_meta_woz_task_classification",
        "task401_numeric_fused_head_reference",
        "task1598_nyc_long_text_generation",
        "task1615_sick_tclassify_b_relation_a",
        "task970_sherliic_causal_relationship",
        "task1390_wscfixed_coreference",
        "task199_mnli_classification",
        "task034_winogrande_question_modification_object",
        "task133_winowhy_reason_plausibility_detection",
        "task226_english_language_answer_relevance_classification",
        "task510_reddit_tifu_title_summarization",
        "task935_defeasible_nli_atomic_classification",
        "task349_squad2.0_answerable_unanswerable_question_classification",
        "task1157_bard_analogical_reasoning_rooms_for_containers",
        "task937_defeasible_nli_social_classification",
        "task743_eurlex_summarization",
        "task1388_cb_entailment",
        "task671_ambigqa_text_generation",
        "task121_zest_text_modification",
        "task1345_glue_qqp_question_paraprashing",
        "task330_gap_answer_generation",
        "task1342_amazon_us_reviews_title",
        "task329_gap_classification",
        "task281_points_of_correspondence",
        "task036_qasc_topic_word_to_generate_related_fact",
        "task1554_scitail_classification",
        "task050_multirc_answerability",
        "task362_spolin_yesand_prompt_response_sub_classification",
        "task1557_jfleg_answer_generation",
        "task249_enhanced_wsc_pronoun_disambiguation",
        "task957_e2e_nlg_text_generation_generate",
        "task418_persent_title_generation",
        "task614_glucose_cause_event_detection",
        "task677_ollie_sentence_answer_generation",
        "task220_rocstories_title_classification",
        "task1631_openpi_answer_generation",
        "task232_iirc_link_number_classification",
        "task1391_winogrande_easy_answer_generation",
        "task1358_xlsum_title_generation",
        "task1533_daily_dialog_formal_classification",
        "task1156_bard_analogical_reasoning_tools",
        "task1659_title_generation",
        "task1624_disfl_qa_question_yesno_classification",
        "task1158_bard_analogical_reasoning_manipulating_items",
        "task827_copa_commonsense_reasoning",
        "task1153_bard_analogical_reasoning_affordance",
        "task393_plausible_result_generation",
        "task879_schema_guided_dstc8_classification",
        "task613_politifact_text_generation",
        "task219_rocstories_title_answer_generation",
        "task190_snli_classification",
        "task200_mnli_entailment_classification",
        "task1534_daily_dialog_question_classification",
        "task1540_parsed_pdfs_summarization",
        "task442_com_qa_paraphrase_question_generation",
        "task392_inverse_causal_relationship",
        "task1562_zest_text_modification",
        "task640_esnli_classification",
        "task1622_disfl_qa_text_modication",
        "task623_ohsumed_yes_no_answer_generation",
        "task020_mctaco_span_based_question",
        "task642_esnli_classification",
        "task102_commongen_sentence_generation",
]


def load_instructions(path, n_tasks=None, ds_limit=None):
    """
    Read all .json files in the directory path and
    extract the field "Definition" from each file.
    """
    import glob
    for i, file in enumerate(glob.glob(path + "/*.json")):   
            task_name = os.path.basename(file).replace(".json", "")
            test_tasks_selected = test_tasks[:n_tasks] if n_tasks else test_tasks
            if task_name in test_tasks_selected: 
                with open(file, "r", encoding="UTF-8") as f:
                    data = json.load(f)  
                # if data["train_examples"]: 
                task_indo =  task_name.split("_")
                task_id = task_indo[0]
                task_category = task_indo[-1]
                if ds_limit is not None:  
                    if i>len(glob.glob(path + "/*.json"))*ds_limit:
                        return task_name, data["Definition"],data["Instances"][:100], data["Positive Examples"]
                yield task_name, data["Definition"],data["Instances"][:100], data["Positive Examples"]


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

@torch.no_grad()                
def generate_outputs(model, examples, tokenizer, instructions_batch,    
                     temperature=0.7, max_output_length=128,      
                     topic_router=None, skill_selector="topic", kmeans=None, tfidf=None, 
                     use_outputs_for_routing=0, outputs=None):
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
    elif kmeans is not None and tfidf is not None: 
        instructions_batch_for_routing=copy.deepcopy(instructions_batch)        
        if use_outputs_for_routing:
            for i, (inst, out) in enumerate(zip(instructions_batch_for_routing, outputs)):
                instructions_batch_for_routing[i]=inst+" "+out[0]
        flag=0
        if len(instructions_batch_for_routing)==1:
            flag=1
            instructions_batch_for_routing=[instructions_batch_for_routing[0],instructions_batch_for_routing[0]]       
        cluster = example_in_cluster(instructions_batch_for_routing,  tfidf, kmeans, random_clusters=False, distances=True)
        if flag:   
            cluster=[cluster[0]]     
        input["distances"] = torch.stack(cluster).cuda()  
    input["pad_token_mask"] = (inputs.input_ids != tokenizer.pad_token_id).float().to(device)
        
    # eval with the best adapter for this cluster
    output_ids = model.generate(
        input,
        # do_sample=True,
        temperature=temperature,      
        max_new_tokens=max_output_length,
        # top_k=50,
        return_dict_in_generate=True,
    )   
     
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
       
def format(task_definition, examples, input):
    out=f"Task definition: {task_definition}\n"
    for example in examples:
        out+=f"Input: {example['input']}\nOutput: {example['output']}\n\n"
    out += f"Input: {input}\nOutput:"
    return out
###############################################################
# this stuff is needed to be in this file to load tfidf vectorizer etc., TODO: clarify why.   
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
###############################################################

def load_kmeans_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        out = pickle.load(f)
    return out
             
def eval_rouge(prediction_file, reference_file="/home/v-oostapenko/dev/mttl/inst_follow/eval/ni/test_references.jsonl", clean=0):    
    from projects.instr_routing.eval.ni.evaluate import compute_metrics, compute_grouped_metrics
    eval_instances = {} 
    with open(reference_file) as fin:
        for line in fin:
            instance = json.loads(line)
            # if track is not provided in the refernce file, we use set the track to `default` and use the default tokenizer in rouge-score.
            if "track" not in instance:
                instance["track"] = "default"
            eval_instances[instance["id"]] = instance

    all_predictions = {} 
    with open(prediction_file) as fin:
        for line in fin:
            prediction = json.loads(line)   
            id = prediction["id"] 
            task = prediction["task_name"]
            # if task in tasks:
            prediction=prediction["prediction"]
            if "Input:" in prediction and clean:
                prediction=prediction.split("Input:")[0]
            all_predictions[id] = prediction.strip()

    all_results = {}
    for track in ["default", "xlingual"]:
        print("Evaluating track:", track)
        instance_ids = [id for id, instance in eval_instances.items() if instance["track"] == track]
        references = [eval_instances[id]["references"] for id in instance_ids]
        predictions = []
        instructions = []
        missing_predictions = []
        for id in instance_ids:
            if id in all_predictions:    
                predictions.append(all_predictions[id])
            else:
                missing_predictions.append(id)
                predictions.append("")
        if missing_predictions:
            print(f"No prediction for {len(missing_predictions)} instances. Use empty string as prediction.")

        results = compute_metrics(predictions, references, xlingual=(track == "xlingual"))
        print("======== Overall Metrics ========")
        for metric, value in results.items():
            print(f"{metric}: {value}")
            all_results[f"{metric}_{track}_track"] = value

        if "task_category" in eval_instances[instance_ids[0]]: 
            categories = ["_".join(eval_instances[id]["task_category"].lower().split()) for id in instance_ids]
            results_per_category = compute_grouped_metrics(predictions, references, categories, xlingual=(track == "xlingual"))
            print("======== Metrics per Category ========")
            for metric, value in results_per_category.items():
                print(f"{metric}: {value}")
                all_results[f"{metric}_{track}_track"] = value

        if "task_id" in eval_instances[instance_ids[0]]:
            tasks = [eval_instances[id]["task_id"] for id in instance_ids]
            results_per_task = compute_grouped_metrics(predictions, references, tasks, xlingual=(track == "xlingual"))
            print("======== Metrics per Task ========")
            for metric, value in results_per_task.items():
                print(f"{metric}: {value}")
                all_results[f"{metric}_{track}_track"] = value
        return all_results
           
@click.command()                    
@click.option("--model_name", type=str, default="alpaca_smear_12_xr4_cos_noxcond") #chavinlo/alpaca-native") yahma/llama-7b-hf chainyo/alpaca-lora-7b togethercomputer/RedPajama-INCITE-Base-7B-v0.1
@click.option("--batch_size", type=int, default=5)  
@click.option("--out_prefix", type=str, default="test")    
@click.option("--from_hf", type=int, default=0)
@click.option("--model_path", type=str, default="") # ~/dev/amlt/alpaca_poly/alpaca_lora_poly_merge_sep/yahma_llama-7b-hfle2xye5c_alpaca_lora_full_r4-val/best_model_alpaca_lora_full_r4_loss=0.4225.ckpt/loss=0.4225.ckpt") #~/logs/amlt_yahma_llama_atlas_cluster_l1/alpaca4r_topic_ldal1/alpaca-lora_l1/best_model_alpaca_lora_atlas_cluster_te_ada_l1/loss=0.4242.ckpt")#"~/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt") #"~/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt") #"~/logs/llama_alpaca/lora_full/yahma_llama-7b-hfopq9a3dw_alpaca_lora_full-val/loss=0.5940.ckpt")
# @click.option("--example_to_ids_path", type=str, default="~/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002.pkl") # depreciated
@click.option("--skill_selector", type=str, default="poly") # average
@click.option("--nshot", type=int, default=0) # >0 means use canonical examples
# @click.option("--reference_file", type=str, default="~/dev/mttl/inst_follow/eval/ni/test_references.jsonl") # >0 means use canonical examples
@click.option("--n_tasks", type=int, default=None) # if None, will use a subset of test tasks
@click.option("--use_wandb", type=int, default=0) # 
@click.option("--use_outputs", type=int, default=0) #     
@click.option("--amlt_experiment_name", type=str, default="alpaca_smear")   
@click.option("--ds_limit", type=float, default=0.01)  
def eval_superni_command(model_name="gpt3", batch_size=4, out_prefix="", from_hf=0, model_path="", 
                         skill_selector="topic", 
                         nshot=0, n_tasks=None, use_wandb=False, use_outputs=False, amlt_experiment_name="alpaca_smear", ds_limit=None):
    return eval_superni(model_name, 
                        batch_size,    
                        out_prefix, from_hf, model_path, skill_selector, nshot, n_tasks, use_wandb, use_outputs, amlt_experiment_name, ds_limit=ds_limit)


def load_model_for_generation(from_hf, base_model_name, model_name, model_path, skill_selector, code_dir):
    topic_router=None
       # correct(data_path, model_name, batch_size, out_prefix, from_hf, model_path, example_to_ids_path, skill_selector, nshot, reference_file)
    if from_hf==1:             
        # if "llama" in model_name:       
        config = {"model": model_name, "model_modifier":None, "example_to_ids_path": None} 
        config = dict_to_dataclass(config)
        model, _, _ = load_model(config, device=device, tokenizer_path="yahma/llama-7b-hf") 
        # tokenizer =  LlamaTokenizer.from_pretrained(model_name, padding_side='left')   
        tokenizer =  LlamaTokenizer.from_pretrained("yahma/llama-7b-hf", padding_side='left')   
        tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
        # tokenizer.padding_side='left'          
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id 
        # else:    
        #     pijama_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)#, device_map="cpu")
        #     tokenizer = AutoTokenizer.from_pretrained(model_name)#.to(device)
        #     tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
        #     tokenizer.padding_side='left'    
        #     config = {"model": model_name, "model_modifier":None} 
        #     config = dict_to_dataclass(config)
        #     model_class = CLM 
        #     config.model_object = pijama_model 
        #     # tokenizer = dm.tokenizer if dm is not None else tokenizer
        #     model = model_class(**vars(config), tokenizer=tokenizer)      
        
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id   
        model.to(device)
    elif from_hf==0:           
        config = Config()       
        config.model = base_model_name
        config.n_skills = 1 # by default, can be overwritten in load_model if a checkpoint is provided
        model, tokenizer, config = load_model(config, model_path, device=device, base_path=code_dir)  
        tokenizer.padding_side='left'  
        print(f"Loaded model {base_model_name} from {model_path}\n")
        print("Loaded config", config.__dict__)
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
    
    # if usepijma_model_with_llama_adapter:    
    #     model.to("cpu")          
    #     pijama_model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1", torch_dtype=torch.float32)#, device_map="cpu")
    #     tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-7B-v0.1")#.to(device)
    #     tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
    #     tokenizer.padding_side='left'            
    #     args = copy.deepcopy(config) #{"adapter_modules": "attention", "model_modifier":"llama_adapter"} 
    #     args.model="togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    #     # args = dict_to_dataclass(args) 
    #     args.adapter_modules = "attention"
    #     pijama_model = modify_transformer(pijama_model, args)
    #     # pijama_model.to("cpu")
    #     # args.lora_modules="v_proj|q_proj|k_proj|o_proj" #"attention"
    #     state_dict = model.model.state_dict()
    #     #load adaption_prompt and adaptation_gate weights into args.model_object
    #     new_state_dict = {}       
    #     for k,n in state_dict.items():   
    #         if "adaption_prompt" in k or "adaption_gate" in k:
    #             #rename key      
    #             ad_layer = k.split(".")[-1]
    #             layer = k.split("layers.")[1].split(".")[0]
    #             # layers.2.attention.adaption_prompt
    #             k = f"gpt_neox.layers.{layer}.attention.{ad_layer}"
    #             new_state_dict[k] = n
    #     pijama_model.load_state_dict(new_state_dict, strict=False)    
    #     assert  int(sum([torch.sum(p) for n,p in model.model.named_parameters() if "adaption" in n]).item()) == int(sum([torch.sum(p) for n,p in pijama_model.named_parameters() if "adaption" in n]).item())
    #     # pijama_model.to(device)
    #     # model.to(device)     
    #     model_class = CLM 
    #     args.model_object = pijama_model 
    #     # tokenizer = dm.tokenizer if dm is not None else tokenizer
    #     model = model_class(**vars(args), tokenizer=tokenizer)
    #     model.to(device)
    #     model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
    #     model.model.config.bos_token_id = tokenizer.bos_token_id
    #     model.model.config.eos_token_id = tokenizer.eos_token_id  
    
    return model, tokenizer, config, topic_router

def eval_superni(model_name="gpt3",  
                 batch_size=4, 
                 out_prefix="", 
                 from_hf=0,                   
                 model_path="", 
                 skill_selector="topic", # almost depreciated 
                 nshot=0, n_tasks=None, use_wandb=0, use_outputs=1, amlt_experiment_name="alpaca_smear",
                 out_dir = "/home/v-oostapenko/out/instr_routing",
                 data_path = "/home/v-oostapenko/dev/natural-instructions/tasks",
                 base_model_path = "/home/v-oostapenko/dev/amlt/",ds_limit=None,
                 ):
    task_results = {}                 
    disable_torch_init()   
    ################################################################################################
    # set paths                
    code_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    base_out = os.getenv("AMLT_OUTPUT_DIR", out_dir)
    if os.environ.get("AMLT_OUTPUT_DIR") is not None: # on gcr
        data_path="/mnt/default/data/natural-instructions/tasks" 
        base_model_path = "/mnt/default/data/models"    
        base_cluster_infos  = "/mnt/default/data/" #/mnt/amlt_code/inst_follow/"
    else:
        base_cluster_infos = code_dir
    path_to_clusterer = f"{base_cluster_infos}/cluster_infos/cbtm/"
    ################################################################################################
        
    if model_path =="" and from_hf==0:
        if model_name in model_dict:    
            # can also list models in the model dict  
            from_hf = model_dict[model_name]["from_hf"]
            model_path = model_dict[model_name]["model_path"]
            out_prefix = model_name
        else:      
            exp_name = amlt_experiment_name
            from_hf = 0
            model_path = glob.glob(f"{base_model_path}/{exp_name}/{model_name}/yahma*/loss=*.ckpt")[0]
            out_prefix = model_name
    
    
          
    base_model_name = "yahma/llama-7b-hf"       
    model, tokenizer, config, topic_router = load_model_for_generation(from_hf, base_model_name, 
                                                                       model_name, model_path, skill_selector, code_dir)
    
             
    tfidf, kmeans = None, None
    # infer what clusterer we are using
    cbtm_n_clusters=-1        
    if config.example_to_ids_path is not None and skill_selector!="average" and config.poly_selector!="x_router":
        if "kmeans" in config.example_to_ids_path:
            cbtm_n_clusters = ClusterResult(config.example_to_ids_path).n_clusters()
            dataset_name = config.dataset
        
    if cbtm_n_clusters>0:
        tdifd = "tfidf"
        clusterer=f"kmeans{cbtm_n_clusters}"      
        if dataset_name == "flan_v2":
            clusterer = f"kmeans_model_flnv2_{cbtm_n_clusters}"
            tdifd = "vectorizer_tfidf_flnv2"
            if "w_outputs" in config.example_to_ids_path:
                clusterer = f"kmeans_model_flnv2_{cbtm_n_clusters}_w_outputs"
                tfidf = "vectorizer_tfidf_flnv2_w_outputs"
        elif dataset_name=="human":
            raise NotImplementedError
            clusterer = f"kmeans_human{cbtm_n_clusters}"
            tdifd = "tfidf_human"
            
        elif dataset_name=="alpaca":
            clusterer = f"kmeans_model_SI_{cbtm_n_clusters}"
            tdifd = "vectorizer_tfidf_SI"
            if "w_outputs" in config.example_to_ids_path:
                tdifd = "vectorizer_tfidf_SI_w_outputs"
                clusterer = f"kmeans_model_SI_{cbtm_n_clusters}_w_outputs"
        tfidf = load_kmeans_model(path_to_clusterer+f"/{tdifd}.pkl")
        kmeans = load_kmeans_model(path_to_clusterer+f"/{clusterer}.pkl")
        topic_router = None
        
    print(config)
    print("Arguments:\n")
    print(data_path, "\n", model_name, "\n", batch_size, "\n", out_prefix, "\n", from_hf, "\n", model_path, "\n", skill_selector)
    # nshot = 1
    
    for nshot in [nshot]:#, 5, 10, 15, 20]:      
        out_file_name = f"ni_pred_{out_prefix}_{model_name}ni-nshot{0}.jsonl" if nshot == 0 else f"ni_pred_{out_prefix}_{model_name}ni-nshot_canonical.jsonl"
        out_file_name=out_file_name.replace("/", "_")  
        out_file_name = out_file_name.strip()
        
        
        task_results_existing=None
        
        # create directory if doesnt exist
        if not os.path.exists(f"{base_out}/eval/ni"):
            # create
            os.makedirs(f"{base_out}/eval/ni")
        
        if os.path.exists(f"{base_out}/eval/ni/{out_file_name}"):
            with open(f"{base_out}/eval/ni/{out_file_name}") as f:
                lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            #unique task names
            task_results_existing = {l["task_name"] for l in lines}
                    
        for inst in load_instructions(data_path, n_tasks, ds_limit): # iterate over tasks
            task_name, definition, test_exs, train_exs = inst
            if task_results_existing is not None and task_name in task_results_existing:
                print(f"Skipping {task_name}")
                continue
            batch = []
            outputs = []

            task_results[task_name] = []
            train_examples = train_exs #[ex for ex in train_exs[:nshot]]
            example_ids=[]
            instructions_batch = []
            for example in tqdm.tqdm(test_exs):                
                task_def = definition[0]
                example_ids.append(example['id'])
                input = example['input']
                if nshot == 0:          
                    examples = []
                else:   
                    examples = train_examples 
                msg = format(task_def, examples, input)
                instructions_batch.append(task_def+input)
                batch.append(msg)  
                outputs.append(example['output'])
                if len(batch) == batch_size:   
                    gens = generate_outputs(model, batch, tokenizer, instructions_batch, 
                                            temperature=0.7, topic_router=topic_router,         
                                            skill_selector=skill_selector, kmeans=kmeans, tfidf=tfidf, use_outputs_for_routing=use_outputs, outputs=outputs)
                    for id, pred in zip(example_ids, gens):
                        task_results[task_name].append({"id": id,"prediction": pred, "task_name": task_name})

                    batch = []
                    outputs = []
                    example_ids = [] 
                    instructions_batch = []

            if len(batch):  
                gens = generate_outputs(model, batch, tokenizer, instructions_batch,
                                        temperature=0.7, topic_router=topic_router,         
                                        skill_selector=skill_selector, kmeans=kmeans, tfidf=tfidf, use_outputs_for_routing=use_outputs, outputs=outputs)
                for id, pred in zip(example_ids, gens):
                        task_results[task_name].append({"id": id,"prediction": pred, "task_name": task_name})
                batch = []
                outputs = []
                example_ids = []    
                instructions_batch = []
            
            with open(f"{base_out}/eval/ni/{out_file_name}", "a") as f:
                for l in task_results[task_name]:
                    f.write(json.dumps(l) + "\n")
                 
        #calculate rouge-L        
        all_results_rouge = eval_rouge(f"{base_out}/eval/ni/{out_file_name}", reference_file=f"{code_dir}/eval/ni/test_references.jsonl")
        print(all_results_rouge)
        out_file_name = "[rouge]"+out_file_name 
        with open(f"{base_out}/eval/ni/{out_file_name}", "a") as f:
            for k,v in all_results_rouge.items():
                f.write(f"{k}: {v}\n")
        task_results=[]  
        
        if use_wandb:   
            import wandb
            wandb_project = os.getenv("WANDB_PROJECT", "alpaca_eval")            
            args = {"data_path": data_path, 
                    "model_name": model_name, 
                    "batch_size": batch_size, 
                    "out_prefix": out_prefix,   
                    "from_hf": from_hf, "model_path": model_path, "skill_selector": skill_selector, "nshot": nshot, "use_outputs": use_outputs}
            wandb.init(project=wandb_project, name=os.environ.get("AMLT_JOB_NAME", out_file_name), config=args)
            wandb.log(all_results_rouge)
        del model
        # clean cash
        torch.cuda.empty_cache()
        del tokenizer
        return all_results_rouge["rougeL_default_track"]
              
if __name__ == '__main__':
    eval_superni_command()