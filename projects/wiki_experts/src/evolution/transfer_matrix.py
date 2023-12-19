import os
import sys
import copy
import wandb
import pandas as pd
import seaborn as sns
from typing import Dict
from huggingface_hub import login
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import (
    ExpertLibrary,
    HFExpertLibrary,
    LocalExpertLibrary,
    remove_outdated_experts_from_library,
)

# from projects.wiki_experts.src.evolution.evolving_expert_library import (
#     EvolvingHFExpertLibrary,
# )
from projects.wiki_experts.src.evolution.config import EvolExpertConfig
from projects.wiki_experts.src.evolution.utils import (
    log_wandb,
    init_wandb_logger,
    TableLogger,
)

from projects.wiki_experts.src.evolution.evaluators import Evaluator, prepare_evaluator
from mttl.utils import setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import MultiExpertModel
from mttl.vllm_engines.engines import free_memory
from mttl.models.modifiers.expert_containers.module_graph import Expert, load_expert

DEBUG = False
if "AMLT_OUTPUT_DIR" in os.environ:
    DEBUG = False
if DEBUG:
    print("!!!!!!!!!!!!!!!!!!!!!! DEBUG MODE")


class TransferMAtrixConfig(EvolExpertConfig):
    def _set_defaults(self):
        super()._set_defaults()
        self.only_diagonal = False


def eval_expert_on_task(
    task,
    module: MultiExpertModel,
    expert,
    evaluator_train=None,
    evaluator_valid=None,
    evaluator_test=None,
):
    logger.info(f"Evaluating perf for {task}")
    if expert is not None:
        model_copy = copy.deepcopy(module)
        if isinstance(expert, str):
            model_copy.load_from_module_dict({task: expert}, action="route")
        elif isinstance(expert, Expert):
            model_copy.add_expert_instance(expert, task, action="route")
        else:
            raise ValueError(f"Checkpoint type {type(expert)} not supported")
        if len(model_copy.experts) == 1:
            model_copy.replace_container_with_expert(task, get_expert_instance=False)
        module = model_copy

    result = {}
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
    base_model: MultiExpertModel,
    expert_lib: dict,
    evaluator: Evaluator = None,
):
    log_row = {}
    for expert_name, expert in expert_lib.items():
        score = eval_expert_on_task(
            task_eval_on, base_model, expert, evaluator_test=evaluator
        )
        log_row[expert_name] = score["test"]
    return log_row


def produce_transfer_matrix(
    args: TransferMAtrixConfig,
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
            args,
            args.dataset,
            tasks=task_eval_on,
            split="test",
        )
        module = MultiExpertModel(
            **vars(args),
            tokenizer=evaluator.datamodule.tokenizer,
            device_map="cpu",
        )
        if args.only_diagonal:
            library = copy.deepcopy(expert_lib)
            for n, expert in library.items():
                if expert.expert_info.expert_task_name != task_eval_on:
                    library.remove_expert(n)
        else:
            library = expert_lib

        log_row_task = eval_all_experts_on_task(
            task_eval_on, module, library, evaluator=evaluator
        )
        log_row.update(log_row_task)
        # eval on base model
        log_row["base"] = eval_expert_on_task(
            task_eval_on, module, None, evaluator_test=evaluator
        )["test"]

        print(transfer_table.df)
        transfer_table.log(log_row)
        transfer_table.log_table_wandb()

    transfer_table.means()
    transfer_table.log_table_wandb()

    transfer_matrix = transfer_table.df
    if wandb.run is not None:
        _size = 1 * len(transfer_matrix.columns)
        plt.figure(figsize=(_size, _size))
        transfer_matrix = transfer_matrix.set_index("eval_task")
        ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
        ax.figure.tight_layout()
        wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
    plt.clf()
    return transfer_matrix


def run_eval(args: EvolExpertConfig, debug=None):
    """
    Create transfer matrix.
    """
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    global DEBUG
    if debug is not None:
        DEBUG = debug

    if not DEBUG:
        if wandb.run is not None:
            init_wandb_logger(args)
    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    print("###### Tasks", args.finetune_task_name)
    # can work with other library types as well, but need to implement clone and filter_with_tasks
    if os.path.isfile(args.hf_repo_id):
        # testing a single model
        expert = load_expert(args.hf_repo_id)
        expert.expert_info.expert_name = "joint"
        expert.expert_info.expert_task_name = "joint"
        temp_dir = TemporaryDirectory(dir=args.output_dir + "/")
        expert_lib = LocalExpertLibrary.from_expet_dict(
            {args.hf_repo_id: expert}, destination=temp_dir.name
        )
    else:
        expert_lib: LocalExpertLibrary = LocalExpertLibrary.from_remote(
            HFExpertLibrary(repo_id=args.hf_repo_id), destination="/tmp"
        )
        remove_outdated_experts_from_library(expert_lib)

    transfer_matrix: pd.DataFrame = produce_transfer_matrix(
        args, expert_lib, tasks=args.finetune_task_name
    )
    print("Transfer matrix", transfer_matrix)
    transfer_matrix.to_csv(os.path.join(args.output_dir, "transfer_matrix.csv"))


if __name__ == "__main__":
    args = TransferMAtrixConfig.parse()
    run_eval(args)
