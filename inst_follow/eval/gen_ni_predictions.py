import json
import os 
import numpy as np
import click
import json
import tqdm
import torch

from mttl.dataloader import ni_metrics   
from transformers import LlamaTokenizer 
from mttl.models.poly import get_selector
from inst_follow.finetune_llama import parse_config, Config
from inst_follow.utils import load_model, TopicRouter
from mttl.cluster_tuning.cluster_reader import ClusterResult
from fastchat.utils import disable_torch_init
device = "cuda" if torch.cuda.is_available() else "cpu"
def dict_to_dataclass(d):
    from dataclasses import make_dataclass
    return make_dataclass("X", d.keys())(**d)

# test_tasks = [
#     "task1159_bard_analogical_reasoning_containers",
#     "task957_e2e_nlg_text_generation_generate",
#     "task1728_web_nlg_data_to_text",
#     "task619_ohsumed_abstract_title_generation",
#     "task034_winogrande_question_modification_object",
#     "task1664_winobias_text_generation",
#     "task1394_meta_woz_task_classification",
#     "task738_perspectrum_classification",
#     "task1615_sick_tclassify_b_relation_a",
#     "task936_defeasible_nli_snli_classification",
#     "task1155_bard_analogical_reasoning_trash_or_treasure",
#     "task620_ohsumed_medical_subject_headings_answer_generation",
#     "task890_gcwd_classification",
#     "task362_spolin_yesand_prompt_response_sub_classification",
#     "task304_numeric_fused_head_resolution",
#     "task1540_parsed_pdfs_summarization",
#     "task648_answer_generation",
#     "task1516_imppres_naturallanguageinference",
#     "task1161_coda19_title_generation",
#     "task614_glucose_cause_event_detection",
# ]

 
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


def load_instructions(path, n_tasks=None):
    """
    Read all .json files in the directory path and
    extract the field "Definition" from each file.
    """
    import glob
    for file in glob.glob(path + "/*.json"):    
            task_name = os.path.basename(file).replace(".json", "")
            test_tasks_selected = test_tasks[:n_tasks] if n_tasks else test_tasks
            if task_name in test_tasks_selected:
                with open(file) as f:
                    data = json.load(f)        
                # if data["train_examples"]: 
                task_indo =  task_name.split("_")
                task_id = task_indo[0]
                task_category = task_indo[-1]
                yield task_name, data["Definition"],data["Instances"][:100], data["Positive Examples"]


@torch.no_grad() 
def generate_outputs(model, examples, tokenizer, temperature=0.7, max_output_length=128, topic_router=None, skill_selector="topic"):
    otuputs_list=[]               
    inputs = tokenizer(examples,
            padding=True,
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
            probs = topic_router(examples)   
            input["distances"] = probs
        
        
    # eval with the best adapter for this cluster
    output_ids = model.generate(
        input,
        temperature=temperature,      
        max_new_tokens=max_output_length,
    )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    output_cleaned = [out.split(ex)[1] for out,ex in zip(outputs, examples)]           
    otuputs_list.extend(output_cleaned)
    del output_ids
    del input  
    return otuputs_list
       
def format(task_definition, examples, input):
    out=f"Task definition: {task_definition}\n"
    for example in examples:
        out+=f"Input: {example['input']}\nOutput: {example['output']}\n\n"
    out += f"Input: {input}\nOutput:"
    return out

def correct(data_path, model_name, batch_size, out_prefix, from_hf, model_path, example_to_ids_path, skill_selector, nshot, reference_file):
    base = "/home/v-oostapenko/dev/mttl/inst_follow"
    for nshot in [nshot]:#, 5, 10, 15, 20]:  
        out_file_name = f"ni_pred_{out_prefix}_{model_name}ni-nshot{0}.jsonl" if nshot == 0 else f"ni_pred_{out_prefix}_{model_name}ni-nshot_canonical.jsonl"
        out_file_name=out_file_name.replace("/", "_")
        with open(f"{base}/eval/ni/{out_file_name}") as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]
        line_idx=0
        for inst in load_instructions(data_path): # iterate over tasks
            task_name, definition, test_exs, train_exs = inst
            for example in tqdm.tqdm(test_exs):
                lines[line_idx]["id"] = example['id']
                line_idx+=1
        out_file_name = f"{base}/eval/ni/{out_file_name}"
        out_file_name = out_file_name.replace(".jsonl", f"_corrected.jsonl")
        with open(out_file_name, "a") as f:
            for line in lines:
                f.write(json.dumps(line)+"\n")
                
@click.command()       
@click.option("--data_path", type=str, default="/home/v-oostapenko/dev/natural-instructions/tasks")   
# @click.option("--config_path", type=str, default="/home/v-oostapenko/dev/mttl/configs/llama/finetune_full_lora.json")
@click.option("--model_name", type=str, default="yahma/llama-7b-hf") #chavinlo/alpaca-native") yahma/llama-7b-hf chainyo/alpaca-lora-7b
@click.option("--batch_size", type=int, default=3)
@click.option("--out_prefix", type=str, default="test")     
@click.option("--from_hf", type=int, default=1)
# @click.option("--nshot", type=int, default=1) # >0 means use canonical examples
@click.option("--model_path", type=str, default="")#"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hf0r2kuwgx_alpaca_lora_full-val/loss=0.5943.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_atlas_cluster_instr/yahma_llama-7b-hfjab096vi_alpaca_lora_atlas_cluster_te_ada-val/loss=0.5989.ckpt") #"/home/v-oostapenko/logs/llama_alpaca/lora_full/yahma_llama-7b-hfopq9a3dw_alpaca_lora_full-val/loss=0.5940.ckpt")
@click.option("--example_to_ids_path", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/atlas_by_instr_text-embedding-ada-002.pkl")
@click.option("--skill_selector", type=str, default="random")
@click.option("--nshot", type=int, default=1) # >0 means use canonical examples
@click.option("--reference_file", type=str, default="/home/v-oostapenko/dev/mttl/inst_follow/eval/ni/test_references.jsonl") # >0 means use canonical examples
@click.option("--n_tasks", type=int, default=None) # if None, will use a subset of test tasks
def main(data_path, model_name="gpt3", batch_size=4, out_prefix="", from_hf=0, model_path="", example_to_ids_path=None, skill_selector="topic", nshot=0, reference_file=None, n_tasks=None):
    task_results = {} 
    topic_router = None
    disable_torch_init()
    
    # correct(data_path, model_name, batch_size, out_prefix, from_hf, model_path, example_to_ids_path, skill_selector, nshot, reference_file)
    if from_hf==1:             
        config = {"model": model_name, "model_modifier":None} 
        config = dict_to_dataclass(config)
        model, _ = load_model(config, device=device) 
        # tokenizer =  LlamaTokenizer.from_pretrained(model_name, padding_side='left')   
        tokenizer =  LlamaTokenizer.from_pretrained("yahma/llama-7b-hf", padding_side='left')   
        tokenizer.pad_token_id = 0 #tokenizer.eos_token_id
        # tokenizer.padding_side='left'      
        
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id   
    elif from_hf==0:           
        config = Config()       
        config.model = model_name
        config.n_skills = 1 # by default, can be overwritten in load_model if a checkpoint is provided
        model, tokenizer = load_model(config, model_path, device=device)  
        tokenizer.padding_side='left'
        print(f"Loaded model {model_name} from {model_path}\n")
        print("Loaded config", config.__dict__)
        if example_to_ids_path is not None:
            cluster_result = ClusterResult(example_to_ids_path)      
        if model.args.n_skills>1:          
            topic_router = TopicRouter(cluster_with='instruction')
            all_topics=topic_router.map.get_topic_data()    
            assert cluster_result is not None, "For soft-clustering models, cluster_result must be provided"
            assert model.args.n_skills == len(all_topics) == cluster_result.n_clusters() 
             
            if skill_selector == "average":                  
                    topic_router = None
                    skill_ids_to_keep = np.where(np.bincount(cluster_result._instance.infos.cluster_ids)>0)[0]
                    model.model.remove_skills(skill_ids_to_keep)
                    model.model.switch_selector_to_average(selector_to_replace=get_selector(config).__class__)
                    model.to(device)      
        
        
        model.model.config.pad_token_id = tokenizer.pad_token_id #= 0  # unk
        model.model.config.bos_token_id = tokenizer.bos_token_id
        model.model.config.eos_token_id = tokenizer.eos_token_id                  
    elif from_hf==3: #use perf        
        from peft import PeftModel    
        from inst_follow.models.clm import CLM  
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
        
    print(config)
    print("Arguments:\n")
    print(data_path, "\n", model_name, "\n", batch_size, "\n", out_prefix, "\n", from_hf, "\n", model_path, "\n", example_to_ids_path, "\n", skill_selector)
    # nshot = 1
    
    for nshot in [nshot]:#, 5, 10, 15, 20]:      
        out_file_name = f"ni_pred_{out_prefix}_{model_name}ni-nshot{0}.jsonl" if nshot == 0 else f"ni_pred_{out_prefix}_{model_name}ni-nshot_canonical.jsonl"
        out_file_name=out_file_name.replace("/", "_")
        base = "/home/v-oostapenko/dev/mttl/inst_follow"
        
        
        task_results_existing=None
        if os.path.exists(f"{base}/eval/ni/{out_file_name}"):
            with open(f"{base}/eval/ni/{out_file_name}") as f:
                lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            #unique task names
            task_results_existing = {l["task_name"] for l in lines}
                    
        for inst in load_instructions(data_path, n_tasks): # iterate over tasks
            task_name, definition, test_exs, train_exs = inst
            if task_results_existing is not None and task_name in task_results_existing:
                print(f"Skipping {task_name}")
                continue
            batch = []
            outputs = []

            task_results[task_name] = []
            train_examples = train_exs #[ex for ex in train_exs[:nshot]]
            example_ids=[]
            for example in tqdm.tqdm(test_exs):                
                task_def = definition[0]
                example_ids.append(example['id'])
                input = example['input']
                if nshot == 0:          
                    examples = []
                else:
                    examples = train_examples
                msg = format(task_def, examples, input)
                batch.append(msg)
                outputs.append(example['output'])
                if len(batch) == batch_size:   
                    gens = generate_outputs(model, batch, tokenizer, temperature=0.7, topic_router=topic_router, skill_selector=skill_selector)
                    for id, pred in zip(example_ids, gens):
                        task_results[task_name].append({"id": id,"prediction": pred, "task_name": task_name})

                    batch = []
                    outputs = []
                    example_ids = []

            if len(batch):  
                gens = generate_outputs(model, batch, tokenizer, temperature=0.7, topic_router=topic_router, skill_selector=skill_selector)
                for id, pred in zip(example_ids, gens):
                        task_results[task_name].append({"id": id,"prediction": pred, "task_name": task_name})
                batch = []
                outputs = []
                example_ids = []
            
            with open(f"{base}/eval/ni/{out_file_name}", "a") as f:
                for l in task_results[task_name]:
                    f.write(json.dumps(l) + "\n")
        task_results=[]
            
if __name__ == '__main__':
    main()