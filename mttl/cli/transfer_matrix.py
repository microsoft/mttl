import copy
import os
import sys
from functools import partial
from tempfile import TemporaryDirectory
from typing import Callable, Union

import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything

import wandb
from mttl.config import Args, EvaluationConfig, ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.evaluators.evaluators import (
    Evaluator,
    ExtendedMMLUEvaluator,
    ExtendedRougeEvaluator,
)
from mttl.logging import TableLogger, init_wandb_logger, logger, setup_logging
from mttl.models.expert_model import ExpertModel
from mttl.models.library.expert import Expert, load_expert
from mttl.models.library.expert_library import ExpertLibrary, LocalExpertLibrary
from mttl.utils import remote_login
from mttl.vllm_engines.engines import free_memory


class TransferMatrixConfig(EvaluationConfig):
    only_diagonal = False
    eval_base = True
    transfer_matrix_split = "test"


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
        )
        log_row[expert_name] = score["test"]
    return log_row


def prepare_evaluator(
    args: Args,
    dataset,
    tasks,
    split=None,
    subsample=-1,
    for_generation=None,
):
    from mttl.callbacks import TestLossEvaluator

    if args.eval_metric == "loss":
        EVAL_CLASS = TestLossEvaluator
        for_generation = for_generation if for_generation is not None else False
    elif args.eval_metric == "rougeL":
        EVAL_CLASS = ExtendedRougeEvaluator
        for_generation = for_generation if for_generation is not None else True
    elif args.eval_metric == "acc":
        assert "mmlu" in dataset
        EVAL_CLASS = ExtendedMMLUEvaluator
        for_generation = for_generation if for_generation is not None else True
    else:
        raise ValueError(f"Unknown eval metric {args.eval_metric}")

    args_copy = copy.deepcopy(args)
    args_copy.dataset = dataset
    args_copy.finetune_task_name = tasks
    args_copy.validation_portion = 0.0
    dm = get_datamodule(args_copy, for_generation=for_generation)

    if split is not None:
        evaluator = EVAL_CLASS(
            datamodule=dm,
            subsample=subsample,
            name=tasks,
            split=split,
            use_vllm=args.use_vllm,
        )
        return evaluator

    return partial(
        EVAL_CLASS,
        datamodule=dm,
        name=tasks,
        use_vllm=args.use_vllm,
    )


def create_transfer_matrix(args, checkpoint):
    config = TransferMatrixConfig()

    for k, v in vars(args).items():
        if k in vars(config):
            setattr(config, k, v)

    config.eval_base = False
    config.eval_metric = "rougeL"

    expert: Expert = load_expert(checkpoint)
    expert.expert_info.expert_name = str(args.finetune_task_name)
    expert.expert_info.expert_task_name = str(args.finetune_task_name)

    temp_dir = TemporaryDirectory()
    destination = temp_dir.name

    LocalExpertLibrary.from_expert_dict({"checkpoint": expert}, destination=destination)

    config.library_id = destination
    config.finetune_task_name = (
        args.finetune_task_name.split(",")
        if not isinstance(args.finetune_task_name, list)
        else args.finetune_task_name
    )

    if len(config.finetune_task_name) < 50:
        run_eval(config, debug=False)

    ########################
    temp_dir.cleanup()


def produce_transfer_matrix(
    args: TransferMatrixConfig,
    expert_lib: ExpertLibrary,
    tasks: list,
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
    args.device_map = "cpu"

    for task_eval_on in tasks:
        log_row = {}
        log_row["eval_task"] = task_eval_on

        evaluator: Evaluator = prepare_evaluator(
            args, args.dataset, tasks=task_eval_on, split=args.transfer_matrix_split
        )
        module = ExpertModel(**vars(args))

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
                task_eval_on, module, expert=None, evaluator_test=evaluator
            )["test"]

        print(transfer_table.df)
        transfer_table.log(log_row)
        transfer_table.log_final_table()
        transfer_table.df.to_csv(os.path.join(args.output_dir, "transfer_matrix.csv"))

    transfer_table.means()
    return transfer_table


def run_eval(args: TransferMatrixConfig, debug=None):
    """
    Given a library_id, create transfer matrix: each expert is evaluated on each other expert's dataset.
    """
    seed_everything(args.seed, workers=True)

    if wandb.run is None:
        init_wandb_logger(args)

    remote_login(token=args.remote_token)

    print("###### Tasks", args.finetune_task_name)
    expert_lib = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        token=args.remote_token,
        destination_id=args.destination_library_id,
    )

    transfer_table: TableLogger = produce_transfer_matrix(
        args, expert_lib, tasks=args.finetune_task_name
    )

    transfer_table.log_final_table()
    transfer_matrix = transfer_table.df
    transfer_matrix = transfer_matrix.set_index("eval_task")

    if wandb.run is not None:
        try:
            _size = 1 * len(transfer_matrix.columns)
            plt.figure(figsize=(_size, _size))
            ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
            ax.figure.tight_layout()
            wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
        except Exception as e:
            print(e)

    plt.clf()
    print("Transfer matrix", transfer_matrix)
    transfer_matrix.to_csv(os.path.join(args.output_dir, "transfer_matrix.csv"))
    return transfer_matrix


if __name__ == "__main__":
    args = TransferMatrixConfig.parse()
    run_eval(args)
