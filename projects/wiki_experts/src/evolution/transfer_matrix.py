import os
import sys
import wandb
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from huggingface_hub import login
from pytorch_lightning import seed_everything


from config import ExpertsMergeConfig
from utils import log_wandb, prepare_evaluator, init_wandb_logger, TableLogger

from evaluators import Evaluator
from projects.wiki_experts.src.expert_library import LocalExpertLibrary
from mttl.utils import setup_logging, logger

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
# register models
from projects.wiki_experts.src.expert_model import MultiExpertModel
from mttl.vllm_engines.engines import free_memory


def produce_transfer_matrix(
    args: ExpertsMergeConfig, expert_lib: LocalExpertLibrary, tasks: list
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
        log_row = {}
        log_row["module"] = task_eval_on

        evaluator: Evaluator = prepare_evaluator(
            args,
            args.dataset_test,
            tasks=task_eval_on,
            split=args.test_split,
        )

        for expert_name in expert_lib.get_experts_for_model(args.model):
            module_dest = expert_lib.get_expert_path(args.model, expert_name)

            result = {c: 0 for c in transfer_table.columns}
            result["module"] = expert_name

            logger.info(f"################# Evaluating {expert_name} on {task_eval_on}")

            module = MultiExpertModel(
                **vars(args),
                tokenizer=evaluator.datamodule.tokenizer,
                device_map="cpu",
            )

            if module_dest:
                graph = f"{expert_name} -> linear({module_dest}:1.0)"
                module.load_from_graph_string(graph, action=args.action)
                if args.action == "route":
                    module.convert_container_to_expert(expert_name)

            scores = evaluator.evaluate(module)

            log_row[expert_name] = scores[expert_name]["mean"]

            all = scores.pop("all")
            log_wandb(scores, f"transfer/{expert_name}")
            logger.info(f"Scores on of {expert_name} for {task_eval_on}: {all['mean']}")

            del module
            free_memory()

        transfer_table.log(log_row)
        transfer_table.log_table_wandb()

    transfer_matrix = transfer_table.df
    if wandb.run is not None:
        _size = 1 * len(transfer_matrix.columns)
        plt.figure(figsize=(_size, _size))
        # set "module" as index
        transfer_matrix = transfer_matrix.set_index("module")
        ax = sns.heatmap(transfer_matrix, annot=True, linewidth=0.5)
        ax.figure.tight_layout()
        wandb.log({"transfer_matrix_heatmap": wandb.Image(ax.get_figure())})
    plt.clf()
    return transfer_matrix


def run_eval(args: ExpertsMergeConfig):
    """
    Create transfer matrix.
    """
    seed_everything(args.seed, workers=True)
    setup_logging(args.output_dir)
    init_wandb_logger(args)
    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    print("###### Tasks", args.finetune_task_name)

    expert_lib = LocalExpertLibrary(model_name=args.model, modules_dir=args.modules_dir)

    transfer_matrix: pd.DataFrame = produce_transfer_matrix(
        args, expert_lib, tasks=args.finetune_task_name
    )
    print("Transfer matrix", transfer_matrix)
    transfer_matrix.to_csv(os.path.join(args.output_dir, "transfer_matrix.csv"))


if __name__ == "__main__":
    args = ExpertsMergeConfig.parse()
    run_eval(args)
