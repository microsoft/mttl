import argparse
import json
import os
import time
import torch
import random
import openai
import tqdm
import ray
import re
import shortuuid
import logging
import numpy as np
from projects.instr_routing import global_vars

# from sklearn.decomposition import PCA
from nomic import AtlasProject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)     
from projects.instr_routing.models.clm import CLM
from peft import prepare_model_for_int8_training
from projects.instr_routing.models.modify_model import modify_transformer
from transformers import LlamaForCausalLM, LlamaTokenizer
from projects.instr_routing.cluster_tuning.cluster_reader import ClusterResult

MAX_API_RETRY = 10
REQ_TIME_GAP = 10


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def get_openai_embedding(input, engine="text-embedding-ada-002"):
    exponential_base: float = 2
    for i in range(MAX_API_RETRY):
        try:
            response = openai.Embedding.create(input=input, engine=engine)
            example_embedding = np.array([r["embedding"] for r in response["data"]])
            return example_embedding
        except Exception as e:
            print(e)
            delay = exponential_base * (1 + random.random())
            time.sleep(delay)  # 5
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"


class TopicRouter:
    def __init__(self, cluster_with, n_skills=None) -> None:
        # self.n_skills = config.n_skills
        print(" Loading atlas project...")
        self.project = AtlasProject(name=f"Alpaca_{cluster_with}")
        self.map = self.project.get_map(name=f"Alpaca_{cluster_with}")
        self.n_skills = len(self.map.get_topic_data()) if n_skills is None else n_skills
        # self.pca = PCA(n_components=2048)
        # embeddings_path = args.embeddings_path+f"/embeddings_of_{args.cluster_with}.pkl"
        # if os.path.exists(embeddings_path):
        #     with open(embeddings_path, 'rb') as f:
        #         embeddings = pickle.load(f)
        # else:
        #     raise ValueError("Embeddings file not found")
        # if train_pca:
        #     self.pca.fit(embeddings["embedings"])
        # del embeddings

    def __call__(self, examples, depth=3):
        bs = len(examples)
        example_embeddings = get_openai_embedding(
            examples, engine="text-embedding-ada-002"
        )
        topics = self.map.vector_search_topics(queries=example_embeddings, depth=depth)[
            "topics"
        ]
        probs = np.zeros((bs, self.n_skills))
        for ex in range(bs):
            for k, v in topics[ex].items():
                probs[ex, int(k) - 1] = v
        return probs


def dict_to_dataclass(d):
    from dataclasses import make_dataclass

    return make_dataclass("X", d.keys())(**d)

def remap_old_hp_names(hp):
    old_names={      
        "normalize_xrouter_weights":"xrouter_normalize_weights",
        "x_router_load_balancing":"xrouter_load_balancing",
        "x_router_x_cond":"xrouter_x_cond",
        "x_routing_option":"xrouting_option",
        "normalize_xrouter_weights": "xrouter_normalize_weights",
        "normalize_xrouter_input": "xrouter_normalize_input",
        "reverse_xrouter_kl": "xrouter_reverse_kl",
        "x_router_init_scale": "xrouter_init_scale",
        "xrouter_kaiming": "xrouter_normal_innit",
        "x_router_sim_metric": "xrouter_sim_metric",        
        "sep_teacher_student": "xrouting_sep_teacher_student",
        "xrouter_pad_token_mask": '',
        "validate_before_start": '',
        "per_token_routing": '',
    }
    for old_name, new_name in old_names.items():
        if old_name in hp:
            if new_name!='':
                hp[new_name]=hp[old_name]
            del hp[old_name]
            
    return hp

          
def load_model(args, model_path=None, device="cuda", 
               tokenizer_path=None,
               base_path="/home/v-oostapenko/dev/mttl/projects/instr_routing/"):
    disable_torch_init()
    if isinstance(args, dict):
        args = dict_to_dataclass(args)
    state_dict = None
    if model_path is not None:
        state_dict = torch.load(model_path)
        if hasattr(args, "update_kwargs"):  # isinstance(args, Config):
            loaded_hps = remap_old_hp_names(state_dict["hyper_parameters"])
            args.update_kwargs(loaded_hps)
    # args.router_selector = "x_router"
    state_dict = (
        {k: v.to(device) for k, v in state_dict["state_dict"].items()}
        if state_dict is not None
        else None
    )  
    # reset paths if needed: mostly relaed to models that use cluster infos...
    for k, v in vars(args).items():
        if isinstance(v, str) and "/compositional_adapters/" in v:
            setattr(args, k, v.replace("compositional_adapters", "inst_follow"))
        if isinstance(v, str) and "/mnt/amlt_code/inst_follow/cluster_infos/" in v:
            setattr(
                args,
                k,
                v.replace(
                    "/mnt/amlt_code/inst_follow/cluster_infos/",
                    "/home/v-oostapenko/dev/mttl/inst_follow/cluster_infos/",
                ),
            )
        if (
            isinstance(v, str)
            and "/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/" in v
        ):
            setattr(
                args,
                k,
                v.replace(
                    "/home/v-oostapenko/dev/mttl/inst_follow/data/cluster_infos/",
                    "/home/v-oostapenko/dev/mttl/inst_follow/cluster_infos/",
                ),
            )  
                   
        v = getattr(args, k)
        ############ 
        # overwrite paths if needed       
        if isinstance(v, str) and "/mnt/amlt_code/inst_follow/" in v:
                setattr(args, k, re.sub(r".*/inst_follow/", base_path, v))
        if isinstance(v, str) and "/mnt/amlt_code/inst_follow/" in v:
                setattr(args, k, re.sub(r".*/instr_routing/", base_path, v))
        v = getattr(args, k)
        # check if on gcr
        # if os.environ.get("AMLT_OUTPUT_DIR") is not None: # set cluster infos to point to the rith folder on gcr
        #     if k=='example_to_ids_path':
        #         setattr(args, k, re.sub(r".*/cluster_infos/", "/mnt/amlt_code/inst_follow/cluster_infos/", v))
        # v = getattr(args, k)
    #############################
    if tokenizer_path is None: 
        tokenizer_path = args.model
    tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_path
    )  # for generation we do not aadd the EOS token!!!
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if args.load_in_8bit:
        model = prepare_model_for_int8_training(model)
    else:
        model = model.to(
            global_vars.PRECISION
        )  # upcast only, we train either on top on 8bit or on top of float16.
    if hasattr(args, "example_to_ids_path"):
        if (
            args.example_to_ids_path and "flv2" in args.example_to_ids_path
        ):  # files were renamed, make ut compatible with older models
            args.example_to_ids_path = args.example_to_ids_path.replace("flv2", "flnv2")
    model = modify_transformer(model, args)

    model_class = CLM
    args.model_object = model
    # tokenizer = dm.tokenizer if dm is not None else tokenizer
    module = model_class(**vars(args), tokenizer=tokenizer)
    if hasattr(args, "prune_unused_loras") and args.prune_unused_loras:
        if args.example_to_ids_path is not None:
            cluster_result = ClusterResult(args.example_to_ids_path)
            # prune unused loras
            # skill_ids_to_keep = np.where(
            #     np.bincount(cluster_result._instance.infos.cluster_ids) > 0
            # )[0]
            skill_ids_to_keep = np.where(
                (np.array(cluster_result._instance.infos.cluster_dists).sum(0)) > 0
            )[0]
            module.model.remove_skills(skill_ids_to_keep)
    if state_dict is not None:
        module.load_state_dict(state_dict)  # , strict=False)
    module.to(device)
    return module, tokenizer, args


@ray.remote(num_cpus=4)
def get_eval(sys_prompt, user_prompt: str, max_tokens: int):
    initial_delay: float = 1
    logging.basicConfig(level=logging.INFO)
    delay = initial_delay
    exponential_base: float = 2
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  #'gpt-4',
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            delay *= exponential_base * (1 + random.random())
            time.sleep(delay)  # 5
    logger.error(f"Failed after {MAX_API_RETRY} retries.")
    return "error"


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            raise Exception("Invalid score pair.")
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def parse_score_with_expert(review):
    try:
        score = review.split("\n")[0]
        score = score.replace(",", " ")
        # score to small letters
        score = score.lower()
        if score.startswith("score:"):
            score = score.split("score:")[1].strip()
        if "/" in score:
            score = score.split("/")[0].strip()
        return float(score)
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return -1


def gen_prompt(reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2):
    # Default to general category (index=0)
    reviewer_idx = 0
    for idx, reviewer in enumerate(reviewer_jsons):
        if reviewer["category"] == cat:
            reviewer_idx = idx
            break
    prompt_id = reviewer_jsons[reviewer_idx]["prompt_id"]
    prompt_json = prompt_jsons[prompt_id - 1]
    assert prompt_json["prompt_id"] == prompt_id

    sys_prompt = prompt_json["system_prompt"]
    prompt_template = prompt_json["prompt_template"]
    defaults = prompt_json["defaults"]
    prompt = prompt_template.format(
        question=ques, answer_1=ans1, answer_2=ans2, **defaults
    )

    return sys_prompt, prompt, reviewer_idx + 1


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("-q", "--question-file")
    parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
    parser.add_argument("-p", "--prompt-file")
    parser.add_argument("-r", "--reviewer-file")
    parser.add_argument("-o", "--output-review-file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    ray.init()

    question_jsons = get_json_list(args.question_file)
    answer1_jsons = get_json_list(args.answer_file_list[0])
    answer2_jsons = get_json_list(args.answer_file_list[1])
    reviewer_jsons = get_json_list(args.reviewer_file)
    prompt_jsons = get_json_list(args.prompt_file)

    # check if # of questions, answers are the same
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)

    handles = []
    review_jsons = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))

    for i in question_idx_list:
        assert (
            answer1_jsons[i]["question_id"]
            == question_jsons[i]["question_id"]
            == answer2_jsons[i]["question_id"]
        )

        ques = question_jsons[i]["text"]
        cat = question_jsons[i]["category"]
        ans1 = answer1_jsons[i]["text"]
        ans2 = answer2_jsons[i]["text"]
        sys_prompt, prompt, reviewer_id = gen_prompt(
            reviewer_jsons, prompt_jsons, cat, ques, ans1, ans2
        )
        review_id = shortuuid.uuid()
        review_jsons.append(
            {
                "review_id": review_id,
                "question_id": question_jsons[i]["question_id"],
                "answer1_id": answer1_jsons[i]["answer_id"],
                "answer2_id": answer2_jsons[i]["answer_id"],
                "reviewer_id": reviewer_id,
                "metadata": {},
            }
        )
        # To avoid the rate limit set by OpenAI
        handles.append(get_eval.remote(sys_prompt, prompt, args.max_tokens))
        logger.info(
            f"Waiting for {REQ_TIME_GAP} seconds before sending the next request."
        )
        time.sleep(REQ_TIME_GAP)

    reviews = ray.get(handles)
    with open(f"{args.output_review_file}", "w") as output_review_file:
        for idx, review in enumerate(reviews):
            scores = parse_score(review)
            review_jsons[idx]["text"] = review
            review_jsons[idx]["score"] = scores
            output_review_file.write(json.dumps(review_jsons[idx]) + "\n")
