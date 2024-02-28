import os
import sys
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from torch import nn
from typing import Dict, Union, Callable
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import (
    ExpertLibrary,
    LocalExpertLibrary,
    get_expert_library,
)

from mttl.models.expert_config import ExpertConfig
from projects.wiki_experts.src.utils.utils import (
    init_wandb_logger,
    TableLogger,
    remove_outdated_experts_from_library,
)

from projects.wiki_experts.src.utils.evaluators import Evaluator, prepare_evaluator
from mttl.utils import remote_login, setup_logging, logger

# register models
from mttl.models.expert_model import ExpertModel
from mttl.vllm_engines.engines import free_memory
from mttl.models.modifiers.expert_containers.expert import Expert, load_expert

DEBUG = False
if "AMLT_OUTPUT_DIR" in os.environ:
    DEBUG = False
if DEBUG:
    print("!!!!!!!!!!!!!!!!!!!!!! DEBUG MODE")


class TransferMatrixConfig(ExpertConfig):
    def _set_defaults(self):
        super()._set_defaults()
        self.only_diagonal = False
        self.eval_base = True
        self.transfer_matrix_split = "test"


def eval_expert_on_task(
    task,
    module_constructor: Union[Callable, ExpertModel],
    expert: Expert,
    evaluator_train=None,
    evaluator_valid=None,
    evaluator_test=None,
    debug=False,
):
    module = None
    logger.info(f"Evaluating perf for {task}")

    if expert is not None:
        model_copy: ExpertModel = (
            module_constructor
            if isinstance(module_constructor, ExpertModel)
            else module_constructor()
        )
        assert all(
            [
                key in model_copy.model.state_dict()
                for key in expert.expert_weights.keys()
            ]
        )
        model_copy.model.load_state_dict(expert.expert_weights, strict=False)
        module = model_copy

    if module is None:
        module = (
            module_constructor
            if isinstance(module_constructor, ExpertModel)
            else module_constructor()
        )

    result = {}
    if debug:
        result["test"] = 0.5
        return result
    if evaluator_train is not None:
        scores_base_train = evaluator_train.evaluate(module)
        result["train"] = scores_base_train[task]["mean"]
    if evaluator_valid is not None:
        score_base_valid = evaluator_valid.evaluate(module)
        result["valid"] = score_base_valid[task]["mean"]
    if evaluator_test is not None:
        score_base_test = evaluator_test.evaluate(module)
        result["test"] = score_base_test[task]["mean"]
    del module
    free_memory()
    return result


def eval_all_experts_on_task(
    task_eval_on,
    module_constructor: Union[Callable, ExpertModel],
    expert_lib,
    evaluator: Evaluator = None,
    only_diagonal=False,
):
    log_row = {}
    for expert_name, expert in expert_lib.items():
        if only_diagonal and expert.expert_info.expert_task_name != task_eval_on:
            continue
        score = eval_expert_on_task(
            task_eval_on,
            module_constructor,
            expert,
            evaluator_test=evaluator,
            debug=DEBUG,
        )
        log_row[expert_name] = score["test"]
    return log_row


def produce_transfer_matrix(
    args: TransferMatrixConfig,
    expert_lib: ExpertLibrary,
    tasks: list,
    subsample=-1,
    module=None,
):
    """
    Eval each module in expert_lib on each subject in subjects.
    """
    # sort tasks to first include tasks for which modules are available
    tasks = [t for t in expert_lib.tasks if t in tasks] + [
        t for t in tasks if t not in expert_lib.tasks
    ]

    transfer_table = TableLogger()

    for task_eval_on in tasks:
        log_row = {}
        log_row["eval_task"] = task_eval_on

        evaluator: Evaluator = prepare_evaluator(
            args, args.dataset, tasks=task_eval_on, split=args.transfer_matrix_split
        )
        module = ExpertModel(
            **vars(args),
            device_map="cpu",
        )

        log_row_task = eval_all_experts_on_task(
            task_eval_on,
            module,
            expert_lib,
            evaluator=evaluator,
            only_diagonal=args.only_diagonal,
        )
        log_row.update(log_row_task)
        if args.eval_base:
            # eval on base model
            log_row["base"] = eval_expert_on_task(
                task_eval_on, module, expert=None, evaluator_test=evaluator, debug=DEBUG
            )["test"]

        print(transfer_table.df)
        transfer_table.log(log_row)
        transfer_table.log_table_wandb()
        transfer_table.df.to_csv(os.path.join(args.output_dir, "transfer_matrix.csv"))

    transfer_table.means()
    transfer_table.log_table_wandb()

    transfer_matrix = transfer_table.df
    if wandb.run is not None:
        try:
            _size = 1 * len(transfer_matrix.columns)
            plt.figure(figsize=(_size, _size))
            transfer_matrix = transfer_matrix.set_index("eval_task")
            ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
            ax.figure.tight_layout()
            wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
        except Exception as e:
            print(e)
    plt.clf()
    return transfer_matrix


def resolve_hf_repo_id(hf_repo_id):
    parts = hf_repo_id.split("/")
    if len(parts) == 3:
        return "/".join(parts[:-1]), parts[-1]
    else:
        return hf_repo_id, None


def run_eval(args: TransferMatrixConfig, debug=None):
    """
    Create transfer matrix.
    """
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    global DEBUG
    if debug is not None:
        DEBUG = debug

    # if not DEBUG:
    if wandb.run is None:
        init_wandb_logger(args)

    remote_login(token=args.remote_token)

    print("###### Tasks", args.finetune_task_name)
    # can work with other library types as well, but need to implement clone and filter_with_tasks

    if args.library_id is None:
        # empty lib, eval only base model
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        destination = temp_dir.name
        expert_lib = LocalExpertLibrary(repo_id=destination)
    elif isinstance(args.library_id, Expert):
        expert: Expert = args.library_id
        expert.expert_info.expert_name = "joint"
        expert.expert_info.expert_task_name = "joint"
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        destination = temp_dir.name
        expert_lib = LocalExpertLibrary(repo_id=destination)
        expert_lib.add_expert(expert)
    elif os.path.isfile(args.library_id):
        # testing a single model
        expert = load_expert(args.library_id)
        expert.expert_info.expert_name = "joint"
        expert.expert_info.expert_task_name = "joint"
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        destination = temp_dir.name
        expert_lib = LocalExpertLibrary.from_expert_dict(
            {args.library_id: expert}, destination=destination
        )
    else:
        destination = args.output_dir + "/library/"
        os.makedirs(destination, exist_ok=True)
        hf_repo_id, expert_name = resolve_hf_repo_id(args.library_id)
        expert_lib: LocalExpertLibrary = LocalExpertLibrary.from_expert_library(
            get_expert_library(repo_id=hf_repo_id), repo_id=destination
        )
        if expert_name is not None:
            for name in list(expert_lib.keys()):
                if name != expert_name:
                    expert_lib.remove_expert(name)

        remove_outdated_experts_from_library(expert_lib)

    transfer_matrix: pd.DataFrame = produce_transfer_matrix(
        args, expert_lib, tasks=args.finetune_task_name
    )
    print("Transfer matrix", transfer_matrix)
    transfer_matrix.to_csv(os.path.join(args.output_dir, "transfer_matrix.csv"))

    if os.path.isdir(destination):
        # remove nonempty dir
        try:
            os.system("rm -rf %s" % destination)
        except:
            pass


if __name__ == "__main__":
    args = TransferMatrixConfig.parse()
    run_eval(args)
