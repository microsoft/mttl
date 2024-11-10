import json
import os
from copy import deepcopy

import torch
from pytorch_lightning import seed_everything

import wandb
from mttl.arguments import EvaluationConfig, ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.logging import TableLogger, logger, setup_logging
from mttl.models.containers.selectors.base import Selector, SelectorConfig
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelConfig,
    MultiExpertModel,
    MultiExpertModelConfig,
)
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

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)

    exclude_phi_tasks = [
        "hellaswag_1_1_0",
        "ai2_arc_ARC_Challenge_1_0_0",
        "ai2_arc_ARC_Easy_1_0_0",
        "piqa_1_0_0",
        "winogrande_1_1_0",
        "bool_q_1_0_0",
        "openbookqa_0_1_0",
    ]

    library = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        token=args.remote_token,
        exclude_selection=exclude_phi_tasks if args.expert_selection is None else None,
        destination_id=args.destination_library_id,
        selection=args.expert_selection,
    )
    an_expert = library[next(iter(library.keys()))]
    base_model = an_expert.expert_info.expert_model
    train_cfg = ExpertConfig.from_dict(an_expert.training_config)

    loading_kwargs = {
        "device_map": args.device_map,
        "precision": args.precision,
    }

    # For starts, always overwrite the following arguments
    for arg_name in [
        "output_dir",
        "eval_metric",
        "remove_phi_eval_tasks",
        "include_task_source",
        "subsample_dev",
        "subsample_test",
        "predict_batch_size",
        "add_eos_to_downstream_targets",
    ]:
        value = getattr(args, arg_name, None)

        if value != getattr(train_cfg, arg_name, None):
            logger.info(f"Overwriting {arg_name} in training config with value {value}")

        setattr(train_cfg, arg_name, value)

    """ Parameter Merging Approaches """
    if args.merge_or_route in ["uniform", "ties"]:
        if args.merge_or_route == "uniform":
            expert = WeightedLinearMerge(WeightedLinearMergeConfig()).transform(library)
        elif args.merge_or_route == "ties":
            cfg = TiesMergeConfig(top_k=args.transform_sparsity)
            expert = TiesMerge(cfg).transform(library)

        model = MultiExpertModel(
            MultiExpertModelConfig(base_model=base_model),
            **loading_kwargs,
        )
        model.add_expert_instance(expert, is_default=True)

    elif args.merge_or_route == "uniform_lora_after_op":
        # Here we merge the LoRA experts after the outer product we cannot really do it
        # with the lib transform, cause this would require storing large matrices in memory
        # Instead we do it with a uniform selector
        from mttl.models.containers.selectors.poly_selector import (
            PolySelectorDirectConfigUniform,
        )

        model = MultiExpertModel.from_pretrained_library(
            library,
            selector_config=PolySelectorDirectConfigUniform(lora_merge_after=True),
            **loading_kwargs,
        )
    elif args.merge_or_route == "base":
        model = ExpertModel(
            ExpertModelConfig(base_model=base_model),
            **loading_kwargs,
        )

    elif args.merge_or_route in ["phatgoose", "arrow", "avg_act"]:
        """Routing Approaches"""
        from mttl.models.containers.selectors import (
            ArrowSelectorConfig,
            AverageActivationSelectorConfig,
            PhatgooseSelectorConfig,
        )

        # compute prototypes if not provided
        if args.merge_or_route == "phatgoose":
            selector_config = PhatgooseSelectorConfig.from_training_config(args)
        elif args.merge_or_route == "arrow":
            selector_config = ArrowSelectorConfig.from_training_config(args)
        elif args.merge_or_route == "avg_act":
            selector_config = AverageActivationSelectorConfig.from_training_config(args)

        # if a specific prototype hash is *not* specified in the config, compute it and store them in the library
        # otherwise, the selector data id will be used to load the prototypes automatically
        if not selector_config.selector_data_id:
            selector_config.selector_data_id = fetch_prototypes(args, library)

        model = MultiExpertModel.from_pretrained_library(
            library,
            selector_config=selector_config,
            **loading_kwargs,
        )

    elif args.merge_or_route == "oracle":
        """TaskNameSelector"""
        from mttl.models.containers.selectors import TaskNameSelectorConfig

        selector_config = TaskNameSelectorConfig.from_training_config(args)

        model = MultiExpertModel.from_pretrained_library(
            library,
            selector_config=selector_config,
            **loading_kwargs,
        )
    else:
        raise ValueError(f"Unknown merge_or_route {args.merge_or_route}")

    metric_logger = Selector.metric_logger

    if wandb.run is None and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "0shot_routing"),
            config=vars(args),
            name=os.environ.get("AMLT_JOB_NAME", None),
        )
        # update config
        wandb.config.update({f"cmd_args_{k}": v for k, v in vars(args).items()})

    if args.pipeline_eval_tasks is None:
        logger.info(
            "`pipeline_eval_tasks` was not set, setting pipeline_eval_tasks='all'..."
        )
        args.pipeline_eval_tasks = "all"

    if args.pipeline_eval_tasks in [
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
        scores = eval_in_distribution(model, train_cfg, tasks)
    elif args.pipeline_eval_tasks in [
        "task1356_xlsum_title_generation",
        "task304_numeric_fused_head_resolution",
        "task202_mnli_contradiction_classification",
        "task035_winogrande_question_modification_person",
        "task614_glucose_cause_event_detection",
        "task362_spolin_yesand_prompt_response_sub_classification",
        "task242_tweetqa_classification",
        "task613_politifact_text_generation",
        "task1728_web_nlg_data_to_text",
        "task1153_bard_analogical_reasoning_affordance",
        "task039_qasc_find_overlapping_words",
        "task1557_jfleg_answer_generation",
    ]:
        logger.info(f"Evaluating SNI with Rouge: task {args.pipeline_eval_tasks}")

        train_cfg.finetune_task_name = args.pipeline_eval_tasks
        train_cfg.pipeline_eval_tasks = None
        train_cfg.predict_batch_size = args.predict_batch_size

        dm_for_gen = get_datamodule(train_cfg, for_generation=True)

        rouge_evaluator = RougeEvaluator(dm_for_gen)
        rouge = rouge_evaluator.evaluate(model, split="test", verbose=False)
        logger.info(f"RougeL: {rouge}")
        if wandb.run is not None:
            if rouge is not None:
                wandb.log({f"downstream/test_rougeL": rouge})

        return
    else:
        if args.pipeline_eval_tasks == "all":
            args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        with torch.no_grad():
            runner: EvaluatorRunner = setup_evaluators(
                model_type=base_model,
                model_family=train_cfg.model_family,
                max_input_length=train_cfg.max_input_length,
                max_output_length=train_cfg.max_output_length,
                predict_batch_size=train_cfg.predict_batch_size,
                truncation_side=train_cfg.truncation_side,
                tasks=args.pipeline_eval_tasks,
                output_path=os.path.join(args.output_dir, "DOWNSTREAM"),
                add_eos_to_targets=args.add_eos_to_downstream_targets,
            )
            scores = runner.run(model)

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
