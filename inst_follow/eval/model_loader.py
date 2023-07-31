from inst_follow.utils import load_model, TopicRouter, disable_torch_init
from mttl.cluster_tuning.cluster_reader import ClusterResult
from mttl.models.poly import get_selector
from transformers import LlamaTokenizer
from peft import PeftModel
from inst_follow.models.clm import CLM
from finetune_llama import Config
from inst_follow.utils import TopicRouter
from mttl.cluster_tuning.cluster_reader import ClusterResult
from inst_follow.utils import disable_torch_init
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def dict_to_dataclass(d):
    from dataclasses import make_dataclass

    return make_dataclass("X", d.keys())(**d)


def load_from_mttl(args):
    disable_torch_init()
    config = Config()
    config.model = "yahma/llama-7b-hf"
    config.n_skills = (
        1  # by default, can be overwritten in load_model if a checkpoint is provided
    )
    module, tokenizer, config = load_model(
        config,
        args.model_name_or_path,
        device=device,
    )

    # initialize the topic router
    topic_router = None

    config.example_to_ids_path = args.example_to_ids_path

    tokenizer.padding_side = "left"
    print(f"Loaded model {config.model} from {args.model_name_or_path}\n")

    if config.example_to_ids_path is not None:
        cluster_result = ClusterResult(config.example_to_ids_path)
        if args.skill_selector == "poly":
            assert module.args.n_skills > 1
            module.model.switch_selector_to_average()
        elif module.args.n_skills > 1:
            topic_router = TopicRouter(cluster_with="instruction")
            all_topics = topic_router.map.get_topic_data()
            assert (
                cluster_result is not None
            ), "For soft-clustering models, cluster_result must be provided"
            assert module.args.n_skills == cluster_result.n_clusters()
            if config.prune_unused_loras:
                # prune unused loras
                # counts = m = np.bincount(cluster_result._instance.infos.cluster_ids)
                skill_ids_to_keep = np.where(
                    np.bincount(cluster_result._instance.infos.cluster_ids) > 0
                )[0]
                module.model.skill_ids_to_keep = skill_ids_to_keep
                # model.model.remove_skills(skill_ids_to_keep)
                cluster_result.remove_skills(skill_ids_to_keep)
            if args.skill_selector == "average":
                topic_router = None
                # skill_ids_to_keep = np.where(np.bincount(cluster_result._instance.infos.cluster_ids)>0)[0]
                # model.model.remove_skills(skill_ids_to_keep)
                module.model.switch_selector_to_average(
                    selector_to_replace=get_selector(config).__class__
                )

    module.model.config.pad_token_id = tokenizer.pad_token_id  # = 0  # unk
    module.model.config.bos_token_id = tokenizer.bos_token_id
    module.model.config.eos_token_id = tokenizer.eos_token_id
    return module.model, tokenizer, topic_router


def load_from_llama():
    config = {"model": "yahma/llama-7b-hf", "model_modifier": None}
    config = dict_to_dataclass(config)
    module, _, _ = load_model(config, device=device, tokenizer_path="yahma/llama-7b-hf")
    tokenizer = LlamaTokenizer.from_pretrained("yahma/llama-7b-hf", padding_side="left")
    tokenizer.pad_token_id = 0  # tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = module.model
    model.config.pad_token_id = tokenizer.pad_token_id  # = 0  # unk
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_from_peft():
    config = {"model": "yahma/llama-7b-hf", "model_modifier": None}
    config = dict_to_dataclass(config)
    model, tokenizer, _ = load_model(config, device=device)
    model = PeftModel.from_pretrained(
        model.model,
        "tloen/alpaca-lora-7b",
        device_map={"": device},
    )
    # tokenizer =  LlamaTokenizer.from_pretrained("yahma/llama-7b-hf", padding_side='left')
    tokenizer.pad_token_id = 0  # tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_class = CLM
    config.model_object = model
    # tokenizer = dm.tokenizer if dm is not None else tokenizer
    module = model_class(**vars(config), tokenizer=tokenizer)
    model = module.model

    model.config.pad_token_id = tokenizer.pad_token_id  # = 0  # unk
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer
