import os
import sys
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from mttl.models.modifiers.expert_containers import (
    add_expert_to_transformer,
)
from mttl.evaluators import MMLUEvaluator
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.ranker.config import RankerConfig
from projects.wiki_experts.src.evolution.utils import (
    log_wandb,
    TableLogger,
)

from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)

from mttl.models.modifiers.expert_containers.expert_library import (
    HFExpertLibrary,
)
from mttl.utils import logger
import json
import copy

# register models
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    MultiExpertModelRanker,
)
from mttl.vllm_engines.engines import free_memory
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.evaluators.loss_evaluator import LossEvaluator


def get_all_tasks_using_single_expert(
    args, expert_lib: HFExpertLibrary, task_name, expert_name
):
    args.finetune_task_name = task_name
    evaluator = MMLUEvaluator(args)

    expert_dump = expert_lib.get_expert(expert_name)

    logger.info(f"################# Evaluating {expert_name} on mmlu")

    module = MultiExpertModel(
        **vars(args),
        tokenizer=evaluator.datamodule.tokenizer,
        # device_map="cpu",
    )

    add_expert_to_transformer(
        module.model,
        expert_dump,
        action="route",
        is_default=False,
    )

    module.replace_container_with_expert(expert_name)
    scores = evaluator.evaluate(module)
    print(scores)


def product_transfer_matrix_loss(
    args: ExpertConfig,
    expert_lib: HFExpertLibrary,
    tasks: list,
    candidate_expert_names: list,
    fout=None,
):
    """
    Eval each module in expert_lib on each subject in subjects.
    """
    # sort tasks to first include tasks for which modules are available
    tasks = [t for t in expert_lib.keys() if t in tasks] + [
        t for t in tasks if t not in expert_lib.keys()
    ]

    transfer_table = TableLogger()

    for task_eval_on in tasks:
        config = FlatMultiTaskConfig(
            dataset=args.dataset,
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=task_eval_on,
            predict_batch_size=args.predict_batch_size,
        )

        data_module = FlatMultiTaskModule(config, for_generation=False)

        evaluator = LossEvaluator(data_module)

        ################# add default expert begin ###############
        module_name = list(expert_lib.keys())[0]
        module_dump = expert_lib[module_name]

        # fill all the weights with zeros
        # deep copy the weights
        module_copy = copy.deepcopy(module_dump)
        weights = module_copy.expert_weights
        for key, value in weights.items():
            value.fill_(0)

        module = MultiExpertModel(
            **vars(args),
            tokenizer=data_module.tokenizer,
            # device_map="cpu",
        )
        expert_name = "default"
        module.add_expert_instance(module_copy, expert_name="default", is_default=True)

        module.replace_container_with_expert(expert_name)

        scores = evaluator.get_loss(module, num_batches=5)

        logger.info(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
        print(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
        fout.write(
            json.dumps(
                {
                    "expert_name": expert_name,
                    "task_eval_on": task_eval_on,
                    "score": scores,
                }
            )
            + "\n"
        )

        del module
        free_memory()
        ################# add default expert end ###############

        for expert_name in candidate_expert_names:
            expert_dump = expert_lib.get_expert(expert_name)

            logger.info(f"################# Evaluating {expert_name} on {task_eval_on}")

            if isinstance(evaluator, MMLUEvaluator):
                module = MultiExpertModel(
                    **vars(args),
                    tokenizer=evaluator.datamodule.tokenizer,
                    # device_map="cpu",
                )
            else:
                module = MultiExpertModel(
                    **vars(args),
                    tokenizer=data_module.tokenizer,
                    # device_map="cpu",
                )
            add_expert_to_transformer(
                module.model,
                expert_dump,
                action="route",
                is_default=False,
            )

            module.replace_container_with_expert(expert_name)

            scores = evaluator.get_loss(module, num_batches=5)

            logger.info(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
            print(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
            fout.write(
                json.dumps(
                    {
                        "expert_name": expert_name,
                        "task_eval_on": task_eval_on,
                        "score": scores,
                    }
                )
                + "\n"
            )
            fout.flush()

            del module
            free_memory()

        print(transfer_table.df)
        transfer_table.log_table_wandb()
    transfer_matrix = transfer_table.df
    if wandb.run is not None:
        _size = 1 * len(transfer_matrix.columns)
        plt.figure(figsize=(_size, _size))
        # set "module" as index
        transfer_matrix = transfer_matrix.set_index("eval_task")
        ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
        ax.figure.tight_layout()
        wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
    plt.clf()
    return transfer_matrix


def produce_transfer_matrix_rouge(
    args: ExpertConfig, expert_lib: HFExpertLibrary, tasks: list
):
    """
    Eval each module in expert_lib on each subject in subjects.
    """
    # sort tasks to first include tasks for which modules are available
    tasks = [t for t in expert_lib.keys() if t in tasks] + [
        t for t in tasks if t not in expert_lib.keys()
    ]

    if not os.path.exists(os.path.join(args.output_dir, "transfer_matrix.jsonl")):
        os.mkdir(args.output_dir)
    fout = open(os.path.join(args.output_dir, "transfer_matrix.jsonl"), "w")

    transfer_table = TableLogger()

    for task_eval_on in tasks:
        config = FlatMultiTaskConfig(
            dataset=args.dataset,
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=task_eval_on,
            predict_batch_size=8,
        )

        data_module = FlatMultiTaskModule(config, for_generation=True)

        evaluator = RougeEvaluator(data_module)

        ################# add default expert ###############
        module_name = list(expert_lib.keys())[0]
        module_dump = expert_lib[module_name]

        expert_name = "default"

        # fill all the weights with zeros
        # deep copy the weights
        weights = copy.deepcopy(module_dump.expert_weights)
        for key, value in weights.items():
            value.fill_(0)

        module = MultiExpertModel(
            **vars(args),
            tokenizer=data_module.tokenizer,
            # device_map="cpu",
        )
        add_expert_to_transformer(
            module.model,
            expert_name,
            module_dump.expert_config,
            weights,
            action="route",
            is_default=True,
        )

        module.replace_container_with_expert(expert_name)

        scores = evaluator.evaluate(module, num_batches=5)
        logger.info(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
        print(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
        fout.write(
            json.dumps(
                {
                    "expert_name": expert_name,
                    "task_eval_on": task_eval_on,
                    "score": scores,
                }
            )
            + "\n"
        )

        del module
        free_memory()
        ################# add default expert ###############
        for expert_name, expert_dump in expert_lib.items():
            module_dest = expert_lib[expert_name]

            logger.info(f"################# Evaluating {expert_name} on {task_eval_on}")

            if isinstance(evaluator, MMLUEvaluator):
                module = MultiExpertModel(
                    **vars(args),
                    tokenizer=evaluator.datamodule.tokenizer,
                    # device_map="cpu",
                )
            else:
                module = MultiExpertModel(
                    **vars(args),
                    tokenizer=data_module.tokenizer,
                    # device_map="cpu",
                )

            if module_dest:
                add_expert_to_transformer(
                    module.model,
                    expert_name,
                    expert_dump.expert_config,
                    expert_dump.expert_weights,
                    action="route",
                    is_default=False,
                )

                module.replace_container_with_expert(expert_name)

            scores = evaluator.evaluate(module, num_batches=5)
            logger.info(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
            print(f"Scores on of {expert_name} for {task_eval_on}: {scores}")
            fout.write(
                json.dumps(
                    {
                        "expert_name": expert_name,
                        "task_eval_on": task_eval_on,
                        "score": scores,
                    }
                )
                + "\n"
            )
            fout.flush()

            del module
            free_memory()

        print(transfer_table.df)
        transfer_table.log_table_wandb()
    transfer_matrix = transfer_table.df
    if wandb.run is not None:
        _size = 1 * len(transfer_matrix.columns)
        plt.figure(figsize=(_size, _size))
        # set "module" as index
        transfer_matrix = transfer_matrix.set_index("eval_task")
        ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
        ax.figure.tight_layout()
        wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
    plt.clf()
    return transfer_matrix


def produce_transfer_matrix_mmlu(
    args: RankerConfig, expert_lib: HFExpertLibrary, tasks: list
):
    """
    Eval each module in expert_lib on each subject in subjects.
    """
    # sort tasks to first include tasks for which modules are available
    tasks = [t for t in expert_lib.keys() if t in tasks] + [
        t for t in tasks if t not in expert_lib.keys()
    ]

    if not os.path.exists(os.path.join(args.output_dir, "transfer_matrix.jsonl")):
        os.mkdir(args.output_dir)
    fout = open(os.path.join(args.output_dir, "transfer_matrix.jsonl"), "w")

    transfer_table = TableLogger()

    for task_eval_on in tasks:
        log_row = {}
        log_row["eval_task"] = task_eval_on
        args.finetune_task_name = task_eval_on
        evaluator = MMLUEvaluator(args)

        ################# add default expert ###############
        module_name = list(expert_lib.keys())[0]
        module_dump = expert_lib[module_name]

        expert_name = "default"

        # fill all the weights with zeros
        # deep copy the weights
        module_copy = copy.deepcopy(module_dump)
        weights = module_copy.expert_weights
        for key, value in weights.items():
            value.fill_(0)
        if isinstance(evaluator, MMLUEvaluator):
            module = MultiExpertModel(
                **vars(args),
                tokenizer=evaluator.datamodule.tokenizer,
                # device_map="cpu",
            )

        add_expert_to_transformer(
            module.model,
            expert_name,
            module_copy,
            action="route",
            is_default=True,
        )

        module.replace_container_with_expert(expert_name)

        scores = evaluator.evaluate(module, subsample=args.subsample)
        log_row[expert_name] = scores[task_eval_on]["mean"]

        all = scores.pop("all")
        log_wandb(scores, f"transfer/{expert_name}")
        logger.info(f"Scores on of {expert_name} for {task_eval_on}: {all['mean']}")
        print(f"Scores on of {expert_name} for {task_eval_on}: {all['mean']}")

        del module
        free_memory()
        ################# add default expert ###############
        for expert_name, expert_dump in expert_lib.items():
            module_dest = expert_lib[expert_name]

            logger.info(f"################# Evaluating {expert_name} on {task_eval_on}")

            if isinstance(evaluator, MMLUEvaluator):
                module = MultiExpertModel(
                    **vars(args),
                    tokenizer=evaluator.datamodule.tokenizer,
                    # device_map="cpu",
                )

            if module_dest:
                add_expert_to_transformer(
                    module.model,
                    expert_name,
                    expert_dump,
                    action="route",
                    is_default=False,
                )

                module.replace_container_with_expert(expert_name)

            scores = evaluator.evaluate(module, subsample=args.subsample)

            log_row[expert_name] = scores[task_eval_on]["mean"]

            all = scores.pop("all")
            log_wandb(scores, f"transfer/{expert_name}")
            logger.info(f"Scores on of {expert_name} for {task_eval_on}: {all['mean']}")
            print(f"Scores on of {expert_name} for {task_eval_on}: {all['mean']}")
            fout.write(
                json.dumps(
                    {
                        "expert_name": expert_name,
                        "task_eval_on": task_eval_on,
                        "score": all["mean"],
                    }
                )
                + "\n"
            )
            fout.flush()

            del module
            free_memory()
        fout.write(json.dumps(log_row) + "\n")
        fout.flush()
        transfer_table.log(log_row)
        transfer_table.log_table_wandb()
    transfer_matrix = transfer_table.df
    print(transfer_matrix)
    transfer_matrix.to_csv(os.path.join(args.output_dir, "transfer_matrix.csv"))
    if wandb.run is not None:
        _size = 1 * len(transfer_matrix.columns)
        plt.figure(figsize=(_size, _size))
        # set "module" as index
        transfer_matrix = transfer_matrix.set_index("eval_task")
        ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
        ax.figure.tight_layout()
        wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
    plt.clf()
    return transfer_matrix


def get_transfer_matrix_by_filter_tasks(args):
    import pandas as pd

    df = pd.read_json("top_5_random_5.jsonl", lines=True)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    fout = open(os.path.join(args.output_dir, "transfer_matrix.jsonl"), "w")

    for i, row in df.iterrows():
        expert_lib = HFExpertLibrary(args.hf_lib_id)
        # filter the experts we used

        # get the transfer matrix
        transfer_matrix: pd.DataFrame = product_transfer_matrix_loss(
            args,
            expert_lib,
            tasks=[row["task"]],
            candidate_expert_names=row["candidate_experts"],
            fout=fout,
        )

        print("Transfer matrix", transfer_matrix)


if __name__ == "__main__":
    args = RankerConfig.parse()
    get_transfer_matrix_by_filter_tasks(args)
    # get_all_tasks_using_single_expert(
    #     args,
    #     HFExpertLibrary(args.hf_lib_id),
    #     task_name="abstract_algebra",
    #     expert_name="adversarial_qa_droberta_tell_what_it_is",
    # )
    # produce_transfer_matrix_rouge(
    #     args, HFExpertLibrary(args.hf_lib_id), [args.finetune_task_name]
    # )

    # product_transfer_matrix_loss(
    #     args, HFExpertLibrary(args.hf_lib_id), [args.finetune_task_name]
    # )
    # produce_transfer_matrix_mmlu(
    #     args, HFExpertLibrary(args.hf_lib_id), args.finetune_task_name.split(",")
    # )
    # get_all_tasks_using_single_expert(
    #     args,
    #     HFExpertLibrary(args.hf_lib_id),
    #     "gem_wiki_lingua_english_en_1_1_0",
    # )
