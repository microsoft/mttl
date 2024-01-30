import os
import sys
import copy
import torch
import wandb
import numpy as np
import seaborn as sns
from dataclasses import replace
from functools import partial
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything
from huggingface_hub import login

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from mttl.models.modifiers.expert_containers.module_graph import Expert, load_expert

from projects.wiki_experts.src.evolution.sequential_evolution import *
from projects.wiki_experts.src.evolution.utils import (
    get_loss,
    init_wandb_logger,
    TableLogger,
    remove_outdated_experts_from_library,
)

from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from mttl.models.modifiers.expert_containers.expert_library import (
    get_best_expert_for_task,
    get_best_expert_for_score,
    LocalExpertLibrary,
    HFExpertLibrary,
    ExpertLibrary,
    Score,
)
from projects.wiki_experts.src.evolution.evaluators import (
    Evaluator,
    prepare_evaluator,
    EvalCallback,
)


from mttl.models.modifiers.expert_containers.module_graph import Expert

from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
    increase_version,
)
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.evolution.experiment_state import ExperimentState
from mttl.vllm_engines.engines import free_memory
from projects.wiki_experts.src.evolution.transfer_matrix import (
    eval_all_experts_on_task,
    eval_expert_on_task,
)

ai = 0
DEBUG = True
if "AMLT_OUTPUT_DIR" in os.environ:
    DEBUG = False
if DEBUG:
    print("!!!!!!!!!!!!!!!!!!!!!! DEBUG MODE")

torch.set_float32_matmul_precision("medium")
a_i = 0
log_prefix = None
wandb_logger = None
log_row = {}
temp_dir = None


def setup(args: EvolExpertConfig):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    global wandb_logger
    wandb_logger = init_wandb_logger(args)

    if os.path.isfile(args.hf_repo_id):
        # testing a single model
        expert = load_expert(args.hf_repo_id)
        expert.expert_info.expert_name = "joint"
        expert.expert_info.expert_task_name = "FLAN_19"
        expert_lib = LocalExpertLibrary.from_expert_dict(
            {args.hf_repo_id: expert}, destination="/tmp"
        )
        expert_lib.ignore_sliced = True
    else:
        if DEBUG:
            global temp_dir
            temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
            local_lib_location = temp_dir.name
        else:
            local_lib_location = os.path.join(args.output_dir, args.hf_repo_id)

        os.makedirs(local_lib_location, exist_ok=True)

        if args.hf_token_hub:
            login(token=args.hf_token_hub)

        expert_lib = LocalExpertLibrary.create_from_remote(
            HFExpertLibrary(args.hf_repo_id), local_lib_location
        )
        expert_lib.ignore_sliced = True
        # make sure we only consider modules of the latest version
        remove_outdated_experts_from_library(expert_lib)

    exper_state = ExperimentState(
        config=args,
        active_iteration=0,
        expert_lib=expert_lib,
        results_table=TableLogger(),
    )
    # dont want to overwrite the exp lib from which we start here for now
    if args.experiment_state_path is not None:
        exper_state.load_from_path(args.experiment_state_path)

    tasks = args.finetune_task_name
    expert_lib = exper_state.state.expert_lib

    print("###### Tasks", tasks)
    return exper_state, tasks


def main(args: EvolExpertConfig):
    exper_state, tasks = setup(args)
    tablelogger, expert_lib, iterations_run = (
        exper_state.state.results_table,
        exper_state.state.expert_lib,
        exper_state.state.active_iteration,
    )
    expert_lib: ExpertLibrary = expert_lib
    module = None
    global a_i, log_prefix, log_row

    for task in tasks:
        print("Evolving on task", task)
        log_row = active_task_iteration(
            args,
            task,
            expert_lib,
            module=module,
            ai=ai,
            update_library=False,
            wandb_logger_local=wandb_logger,
        )
        tablelogger.log(log_row)
        tablelogger.log_table_wandb()


if __name__ == "__main__":
    args = EvolExpertConfig.parse()
    main(args)
