import glob
import sys
import os
import numpy as np
from string import Template

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src import mmlu_subject_configs
from projects.wiki_experts.experts_merge.local_modules import base_dir_tempalte
from mttl.utils import logger


def get_module_graph(module_graph):
    if isinstance(module_graph, dict):
        s = ""
        tasks_to_module = {}
        for subject, dest in module_graph.items():
            tasks_to_module[subject] = dest
            s += f"{subject} -> linear({dest}:$weight_{subject});"
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
        tasks_to_module[subject] = file
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
