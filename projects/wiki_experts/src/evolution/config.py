import os
import sys

from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from projects.wiki_experts.src.config import ExpertConfig
from mttl.models.modifiers.expert_containers.module_graph import ExpertInfo
import mttl.datamodule.flan_tasks
import re


def find_version(s):
    match = re.search(r"_v(\d+)$", s)
    return int(match.group(1)) if match else 0


def increase_version(s):
    version = find_version(s)
    if version == 0:
        return f"{s}_v1"
    else:
        name = s.split(f"_v{version}")[0]
        return f"{name}_v{version+1}"


class EvolExpertConfig(ExpertConfig):
    def _set_defaults(self):
        super()._set_defaults()

        ### evolution
        self.action = "route"
        self.init_router_best = False
        self.subsample_ng_train_set = -1
        self.use_vllm = False
        self.regularizer_factor = 0.0
        self.n_ng_iterations = 2
        self.n_active_iterations = 1
        ##########################
        self.new_module_action = "replace"  # or add, None
        self.eval_metric = "rougeL"  # acc , loss

        self.modules_dir = os.environ.get("MODULES_DIR", "amlt/")
        self.finetune_new_expert = False
        self.experiment_state_path = None
        self.evol_expert_routing = "nevergrad"  # sgd
        # retrieval
        self.sk = -1
        self.retrieve_with = "lora_sim"  # random, lora_sim, loss, rougeL
        self.upload_lib_to_hub = False

    def post_init(self):
        super().post_init()
        if self.finetune_task_name is not None:
            self.finetune_task_name = getattr(
                mttl.datamodule.flan_tasks,
                self.finetune_task_name,
                self.finetune_task_name,
            )
