import json
import os
from copy import deepcopy

import torch
import wandb
from pytorch_lightning import seed_everything

from mttl.arguments import EvaluationConfig
from mttl.datamodule.base import get_datamodule
from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.logging import TableLogger, logger, setup_logging
from mttl.models.containers.selectors.base import Selector, SelectorConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    TiesMerge,
    TiesMergeConfig,
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
)
from mttl.models.lightning.callbacks import LossCallback
from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
from mttl.models.modifiers.lora import LoRAConfig
from mttl.utils import remote_login
from mttl.models.expert_model import MultiExpertModelConfig


def eval_in_distribution(module, args: EvaluationConfig, tasks: list):
    args.include_task_source = "*"
    transfer_table = TableLogger()

    for i, task in enumerate(tasks):
        args.finetune_task_name = task
        args.predict_batch_size = 16
        if args.eval_metric in ["val_loss", "loss"]:
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.val_dataloader(), output_dir=args.output_dir, name=task + "_val"
            )
            metric = evaluator.test(pl_module=module).item()

        elif args.eval_metric == "test_loss":
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.test_dataloader(), output_dir=args.output_dir, name=task + "_test"
            )
            metric = evaluator.test(pl_module=module).item()
        elif args.eval_metric == "val_rougeL":
            dm = get_datamodule(args, for_generation=True)
            evaluator = RougeEvaluator(
                datamodule=dm,
            )
            metric = evaluator.evaluate(
                module,
                split="val",
                verbose=False,
            )
        elif args.eval_metric == "rougeL":
            dm = get_datamodule(args, for_generation=True)
            evaluator = RougeEvaluator(
                datamodule=dm,
            )
            metric = evaluator.evaluate(
                module,
                split="test",
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown eval metric {args.eval_metric}")
        if wandb.run is not None:
            wandb.log({f"test/{args.eval_metric}_{task}": metric})
        transfer_table.log({"task": task, args.eval_metric: metric})

    if wandb.run is not None:
        wandb.log(
            {f"mean_{args.eval_metric}": transfer_table.df[args.eval_metric].mean()}
        )

    transfer_table.log(
        {
            "task": "mean",
            args.eval_metric: transfer_table.df[args.eval_metric].mean(),
        }
    )
    transfer_table.log_final_table()


def fetch_prototypes(args: EvaluationConfig, library: ExpertLibrary) -> str:
    """Returns the unique hash storing the saved prototypes."""
    if args.merge_or_route == "phatgoose":
        from mttl.models.containers.selectors.phatgoose_selector import (
            compute_phatgoose_embeddings,
        )

        return compute_phatgoose_embeddings(
            library,
            selector_data_id=args.selector_data_id,
            n_steps_pg=args.n_steps_pg,
            learning_rate_pg=args.learning_rate_pg,
            recompute_prototypes=args.recompute_prototypes,
            default_args=args,
        )
    elif args.merge_or_route == "arrow":
        from mttl.models.containers.selectors.arrow_selector import (
            compute_arrow_embeddings,
        )

        return compute_arrow_embeddings(
            library,
            selector_data_id=args.selector_data_id,
            ab_only=args.ab_only,
            tie_params=args.tie_params,
            tie_op=args.tie_op,
            add_base_proto=args.add_base_proto,
            recompute_prototypes=args.recompute_prototypes,
        )
    elif args.merge_or_route == "hidden":
        from mttl.models.containers.selectors.average_activation_selector import (
            compute_hidden_states,
        )

        return compute_hidden_states(
            library,
            selector_data_id=args.selector_data_id,
            use_base_model_only=args.use_base_model_only,
            max_samples_per_task=args.max_samples_per_task,
            recompute_prototypes=args.recompute_prototypes,
            track=args.track,
            pool=args.pool,
            default_args=args,
        )
    else:
        raise ValueError(f"Unknown merge_or_route {args.merge_or_route}")


def run_eval(args: EvaluationConfig):
    seed_everything(args.seed, workers=True)
    from mttl.models.lightning.expert_module import ExpertModule

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))
    


    # exclude_phi_tasks = [
    #     "hellaswag_1_1_0",
    #     "ai2_arc_ARC_Challenge_1_0_0",
    #     "ai2_arc_ARC_Easy_1_0_0",
    #     "piqa_1_0_0",
    #     "winogrande_1_1_0",
    #     "bool_q_1_0_0",
    #     "openbookqa_0_1_0",
    # ]

    # library = ExpertLibrary.get_expert_library(
    #     repo_id=args.library_id,
    #     token=args.remote_token,
    #     exclude_selection=exclude_phi_tasks if args.expert_selection is None else None,
    #     destination_id=args.destination_library_id,
    #     selection=args.expert_selection,
    # )
    # an_expert = library[next(iter(library.keys()))]
    # train_cfg = deepcopy(an_expert.training_config)
    # train_cfg.subsample_dev = args.subsample_dev
    # train_cfg.subsample_test = args.subsample_test

    # For starts, always overwrite the following arguments
    # for arg_name in [
    #     "output_dir",
    #     "eval_metric",
    #     "remove_phi_eval_tasks",
    #     "include_task_source",
    # ]:
    #     value = getattr(args, arg_name, None)
    #     setattr(train_cfg, arg_name, value)
    module = ExpertModule(**vars(args))

    if wandb.run is None and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "0shot_routing"),
            config=dict(module.hparams),
            name=os.environ.get("AMLT_JOB_NAME", None),
        )
        # update config
        wandb.config.update({f"cmd_args_{k}": v for k, v in vars(args).items()})

    ## if the base model
    if args.merge_or_route == "base":
        train_cfg = EvaluationConfig(
            model=args.model,
            dataset="sordonia/flan-10k-flat",
            eval_metric="rougeL",
            subsample_test=args.subsample_test,
        )

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
            module.load_state_dict(checkpoint)
        tasks = args.finetune_task_name.split(",")
        scores = eval_in_distribution(module.model, train_cfg, tasks)
    elif args.pipeline_eval_tasks in [
        "in_distribution",
    ]:
        tasks = [expert.expert_task_name for expert in library.data.values()]
        tasks = [expert.expert_task_name for expert in library.data.values()]
        if tasks[0] is None:
            # for some older version of lib (in case of joint experts) no expert_task_name was set
            tasks = json.load(open(args.flan_tasks_path))["flan256"]
        # make sure we evaluate each task seperately (so the mean is over tasks at the end)
        tasks = ",".join(tasks).split(",")
        train_cfg.eval_metric = args.eval_metric
        scores = eval_in_distribution(module, train_cfg, tasks)

        with torch.no_grad():
            runner: EvaluatorRunner = setup_evaluators(
                model_type=module.hparams.model,
                model_family=module.hparams.model_family,
                max_input_length=module.hparams.max_input_length,
                max_output_length=module.hparams.max_output_length,
                predict_batch_size=args.predict_batch_size,
                truncation_side=module.hparams.truncation_side,
                tasks=args.pipeline_eval_tasks,
                output_path=os.path.join(args.output_dir, "DOWNSTREAM"),
                add_eos_to_targets=args.add_eos_to_downstream_targets,
            )
            scores = runner.run(module)

    metric_logger = Selector.metric_logger

    if len(metric_logger) > 0:
        task_table = metric_logger.pretty_table(match_on="task|.*uniform.*")
        layer_table = metric_logger.pretty_table(match_on="layer|.*uniform.*")
        expert_p = metric_logger.pretty_table(match_on=".*expert_p|.*uniform.*")
        angle = metric_logger.pretty_table(match_on=".*angle.*")
        print(task_table)
        print(layer_table)
        print(expert_p)
        print(angle)

    if wandb.run is not None:
        if scores is not None:
            wandb.log({f"downstream/{k}": v for k, v in scores.items()})
        if len(metric_logger) > 0:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        wandb.finish()


if __name__ == "__main__":
    args = EvaluationConfig.parse()
    run_eval(args)
