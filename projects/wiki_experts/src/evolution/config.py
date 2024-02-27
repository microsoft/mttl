import os
import sys

from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from projects.wiki_experts.src.config import ExpertConfig
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
        self.to_repo_id: str = None

        self.evolution_warmup_steps = 0
        self.evol_n_eval_times = 10

        self.use_only_modules_for_tasks = (
            False  # if true, only use modules for the given task set for evolution
        )

        self.transfer_matrix_split = "test"

    def post_init(self, silent=False):
        super().post_init(silent=silent)
        if isinstance(self.finetune_task_name, str):
            self.finetune_task_name = self.finetune_task_name.split(",")

    @property
    def __key(self):
        return (
            self.eval_metric,
            self.retrieve_with,
            self.sk,
            self.finetune_task_name,
            self.evol_expert_routing,
        )

    @property
    def hash(self):
        return str.join("_", map(str, self.__key))
