import os
import sys
import glob
import copy
import torch
import numpy as np
from huggingface_hub import login
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from string import Template
from src import mmlu_subject_configs
from mttl.evaluators import MMLUEvaluator
from mttl.utils import setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.config import ExpertConfig
from mttl.datamodule.mmlu_data_module import MMLUDataModule
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory

base_dir = os.environ.get(
    "MODULES_DIR", "/home/v-oostapenko/dev/amlt/wiki_experts_train_per_cat_2_if/"
)
base_dir_tempalte = (
    lambda subject: f"{base_dir}/ll2_13b_expert_2_{subject}__qa-ostapeno_qa-openai_icl5_clen128_maxD-1_maxC8000_0_length_matched___5e-5_/{subject}/meta-llama_Llama-2-13b-hf-mmlu_test_oracle"
)


# MMLU_MODULES={
#     "formal_logic":
#     "machine_learning":
#     "global_facts":
#     "abstract_algebra":
#     "high_school_physics":
#     "college_biology":
#     "high_school_government_and_politics":
#     "prehistory":
#     "security_studies":
#     "sociology":
# }


def get_module_gaph(module_graph):
    if isinstance(module_graph, dict):
        s = ""
        tasks_to_module = {}
        for subject, mapping in module_graph.items():
            tasks_to_module[subject] = Template(mapping)
            s += mapping
        return s, tasks_to_module
    else:
        if module_graph in ["SUB_10"]:
            tasks = getattr(mmlu_subject_configs, module_graph)
        else:
            return module_graph

    s = ""
    tasks_to_module = {}
    for i, subject in enumerate(tasks):
        subject_dit = base_dir_tempalte(subject)
        files = glob.glob(f"{subject_dit}/*.ckpt")
        if len(files) == 0:
            logger.warning(f"no ckpt files found for {subject}")
            continue
        best_idx = np.argmax([int(f.split("/")[-1].split(".")[0]) for f in files])
        file = files[best_idx]

        mapping = f"{subject} -> linear({file}:$weight_{subject});"
        tasks_to_module[subject] = Template(mapping)
        s += mapping
    return s, tasks_to_module


def parse_experts_to_load(experts_to_load):
    kwargs = []

    def find_experts(path):
        import glob

        for path in glob.glob(expert_path + "/**/csv_metrics/", recursive=True):
            yield "/".join(path.split("/")[:-2])

    if type(experts_to_load) != list:
        experts_to_load = [experts_to_load]

    for expert in experts_to_load:
        options = expert.split(":")
        expert_path = options[0]
        expert_path, _, expert_name = expert_path.partition("=")
        all_paths = list(find_experts(expert_path)) or [expert_path]

        if not expert_name:
            expert_name = None

        if len(options) >= 2:
            action = options[1]
        else:
            action = "route"

        if len(options) >= 3:
            load_only_layers = options[2]
        else:
            load_only_layers = None

        is_default = "*" in action
        action = action.replace("*", "")

        if len(all_paths) > 1:
            if is_default:
                raise ValueError(
                    "Cannot define more than one default expert! Are you using * in expert path?"
                )
            if expert_name:
                raise ValueError(
                    "Cannot declare a name when using a wildcard in the expert path!"
                )

        kwargs.append(
            {
                "expert_path": expert_path,
                "action": action,
                "is_default": is_default,
                "expert_name": expert_name,
                "load_only_layers": load_only_layers,
            }
        )
    return kwargs


def run_eval(args):
    seed_everything(args.seed, workers=True)
    # get directory of the current file
    setup_logging(args.output_dir)
    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    module_graph, tasks_to_module = get_module_gaph(args.module_graph)
    print(module_graph)

    # We can do:
    #   - in-distribution evaluation: test sets we consider are the test sets of the tasks we have experts for
    #   - out-of-distribution evaluation: new task

    # Questions:
    # 1. How good is the merging optimization procedure?
    # On a the in-domain val-set of one of the modules in the population, can it converge to the right routing? (run this for each of the 10 test sets)
    # Does it attain perofrmance like the in-domain module? Could it find this module? if not, did it find a better combination?
    # How does it compare to join-training?

    # 2. How well can we generalize to new task? the baseline here is using jointly pre-trained model vs. merging the experts
    # If I could now tain on the new task a bit, is it bette to use as innitialization the merged pexpert vs. jointl pre-trained?

    # Given the modules lets first eval all of them on each other's test sets -> get a tansfe matix
    #

    for subject_to_eval, graph_to_eval in tasks_to_module.items():
        for subject_eval_on, _ in tasks_to_module.items():
            # select dataloader
            graph = graph_to_eval.substitute({f"weight_{subject_to_eval}": 1.0})
            config_copy = copy.deepcopy(args)
            config_copy.finetune_task_name = subject_eval_on
            mmlu = MMLUEvaluator(
                config_copy, split=config_copy.mmlu_test_split, use_vllm=True
            )
            module = MultiExpertModel(
                **vars(config_copy), tokenizer=mmlu.datamodule.tokenizer
            )
            # module.to("cuda")
            module.load_from_graph_string(graph, action="merge")
            scores = mmlu.evaluate(module)
            del module, mmlu
            free_memory()
            print(scores)

    # then for each of the in-domain datasets, we optimize the merging procedure and see if we can get the right routing


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
