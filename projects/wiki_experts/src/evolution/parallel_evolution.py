import os
import sys
import re
import requests
from typing import Dict
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from projects.wiki_experts.src.evolution.utils import (
    get_loss,
    init_wandb_logger,
    TableLogger,
)

from mttl.models.modifiers.expert_containers.expert_library import (
    LocalExpertLibrary,
    HFExpertLibrary,
    ExpertLibrary,
    Score,
    get_expert_library,
)

from projects.wiki_experts.src.evolution.config import (
    EvolExpertConfig,
    increase_version,
)
from mttl.utils import remote_login, setup_logging, logger
from projects.wiki_experts.src.evolution.experiment_state import ExperimentState
from projects.wiki_experts.src.evolution.sequential_evolution import *
from huggingface_hub import create_repo, HfApi
from mttl.models.modifiers.expert_containers.library_transforms import (
    SVDEmbeddingTransform,
    SVDEmbeddingTransformConfig,
)
from projects.wiki_experts.src.evolution.evaluators import MMLUEvalCallback

# this script evolves a single task for 1 active iteration and commits it to the library

ai = 0
DEBUG = True
if "AMLT_OUTPUT_DIR" in os.environ:
    DEBUG = False
if DEBUG:
    print("!!!!!!!!!!!!!!!!!!!!!! DEBUG MODE")


def find_ai(s):
    match = re.search(r"_ai(\d+)$", s)
    return int(match.group(1)) if match else 0


def increase_ai(s):
    ai = find_ai(s)
    if ai == 0:
        return f"{s}_ai1"
    else:
        name = s.split(f"_ai{ai}")[0]
        return f"{name}_v{ai+1}"


def setup(args: EvolExpertConfig):
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    args.n_active_iterations = 1
    global wandb_logger, ai
    token = os.environ.get("HF_TOKEN", args.remote_token)

    remote_login(token=token)
    user_name = HfApi().whoami(token=token)["name"]
    ai = find_ai(args.hf_repo_id)
    if args.to_repo_id is None:
        args.to_repo_id = increase_ai(args.hf_repo_id)
        args.to_repo_id = f"{user_name}/{args.to_repo_id.split('/')[-1]}"
    args.to_repo_id += "_debug" if DEBUG else ""
    create_repo(args.to_repo_id, token=token, exist_ok=True)

    if not DEBUG:
        wandb_logger = init_wandb_logger(args)
        local_lib_location = os.path.join(args.output_dir, args.to_repo_id)
    else:
        global temp_dir
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        local_lib_location = temp_dir.name

    remote_lib = get_expert_library(args.hf_repo_id)
    os.makedirs(local_lib_location, exist_ok=True)
    expert_lib = LocalExpertLibrary.create_from_remote(
        remote_lib, destination=local_lib_location
    )

    expert_lib.ignore_sliced = True
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
    if not isinstance(tasks, list):
        tasks = tasks.split(",")
    expert_lib = exper_state.state.expert_lib
    # remove tasks for which we dont have experts
    # tasks = [t for t in tasks if t in expert_lib.tasks]

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

    callbacks = [partial(MMLUEvalCallback, split="test")]

    for task in tasks:
        # pull updates from remote
        expert_lib.update_from_remote(args.to_repo_id)
        # recalculate embeddings just in case (its fast)
        svd_embedder = SVDEmbeddingTransform(
            SVDEmbeddingTransformConfig(sparsity_threshold=0.5)
        )
        svd_embedder.transform(expert_lib, upload_to_hf=True)

        print("Evolving on task", task)
        log_row: Dict = active_task_iteration(
            args, task, expert_lib, module=module, ai=ai, callbacks=callbacks
        )
        tablelogger.log(log_row)
        tablelogger.log_table_wandb()

        lib_cynced = False
        try:
            # save the expert lib, send updates to remote
            remote_lib = HFExpertLibrary.from_local(
                expert_lib,
                args.to_repo_id,
                force=True,
                upload_aux_data=True,
                only_tasks=tasks,
            )
            logger.info(f"Done, saving to repo {args.to_repo_id}")
            lib_cynced = True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.info(
                    "Too many requests, contin uing without syncyng the library to repote."
                )

    if not lib_cynced:
        # try hard to cync the library
        from mttl.models.modifiers.expert_containers.expert_library import retry

        @retry(max_retries=20, wait_seconds=60 * 10)
        def retry_sync():
            remote_lib = HFExpertLibrary.from_local(
                expert_lib,
                args.to_repo_id,
                force=True,
                upload_aux_data=True,
                only_tasks=tasks,
            )

        retry_sync()


if __name__ == "__main__":
    args = EvolExpertConfig.parse()
    main(args)
