import os
import sys
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from pytorch_lightning import seed_everything
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.expert_containers.selectors import (
    PerTokenSelector,
    Selector,
    SelectorConfig,
)
from mttl.models.modifiers.lora import LoRAConfig

from mttl.utils import logger, remote_login, setup_logging
from mttl.models.expert_model import MultiExpertModel, ExpertModel
from mttl.models.expert_config import ExpertConfig

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.models.modifiers.expert_containers.library_transforms import (
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
    HiddenStateComputer,
    HiddenStateComputerConfig,
    TiesMerge,
    TiesMergeConfig,
    ArrowTransform,
    ArrowConfig,
    PhatgooseTransform,
    PhatgooseConfig,
)

from mttl.callbacks import LossCallback
from mttl.datamodule.base import get_datamodule
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from projects.modular_llm.src.utils.utils import TableLogger


def get_hidden_states(library, args):
    cfg = HiddenStateComputerConfig(
        use_base_model_only=args.use_base_model_only,
        max_samples_per_task=args.max_samples_per_task,
        name=args.expert_embeds_save_name,
        track=args.track,
        pool=args.pool,
    )
    output = HiddenStateComputer(cfg).transform(
        library, recompute=args.recompute_prototypes, default_args=args
    )

    return output


def get_arrow_embeddings(library, args):
    cfg = ArrowConfig(
        name=args.expert_embeds_save_name,
        ab_only=args.ab_only,
        tie_params=args.tie_params or "default",
        tie_op=args.tie_op,
    )
    return ArrowTransform(cfg).transform(
        library,
        recompute=args.recompute_prototypes,
        add_base_proto=args.base_model_proto,
    )


def get_phatgoose_embeddings(library, args):
    phatgoose_transform = PhatgooseTransform(
        PhatgooseConfig(
            n_steps=args.n_steps_pg,
            learning_rate=args.learning_rate_pg,
            name=args.expert_embeds_save_name,
        )
    )
    return phatgoose_transform.transform(
        library, default_args=args, recompute=args.recompute_prototypes
    )


def patch_prototypes(module, library, args, proto_inits=None):
    if not proto_inits and args.router_selector == "arrow_router":
        proto_inits = get_arrow_embeddings(library, args)
    elif not proto_inits and args.router_selector == "avg_act_router":
        proto_inits = get_hidden_states(library, args)
    elif not proto_inits and args.router_selector == "phatgoose_router":
        proto_inits = get_phatgoose_embeddings(library, args)

    for mod in module.modules():
        if isinstance(mod, PerTokenSelector):
            prototypes = []
            patched_layer_name = mod.layer_name.replace(".selector", "")
            for expert_name in mod.expert_names:
                layer_names = proto_inits[expert_name].keys()
                valid_layer_names = [
                    k
                    for k in layer_names
                    if patched_layer_name in k  # k.startswith(patched_layer_name)
                ]
                assert len(valid_layer_names) <= 2, breakpoint()
                key = sorted(valid_layer_names)[0]
                proto = proto_inits[expert_name][key]
                if isinstance(proto, np.ndarray):
                    proto = torch.from_numpy(proto)

                prototypes += [proto.squeeze()]

            logger.info(
                f"setting prototypes for selector at {mod.layer_name} with hidden states from {key}"
            )
            prototypes = torch.stack(prototypes)
            mod.overwrite_prototypes(prototypes)


def eval_in_distribution(module, args: ExpertConfig, tasks: list):
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


def run_eval(args: ExpertConfig):
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
    train_cfg = deepcopy(an_expert.training_config)
    train_cfg.device_map = "cpu"
    train_cfg.subsample_dev = args.subsample_dev
    train_cfg.subsample_test = args.subsample_test

    # For starts, always overwrite the following arguments
    for arg_name in [
        "output_dir",
        "eval_metric",
        "remove_phi_eval_tasks",
        "include_task_source",
    ]:
        value = getattr(args, arg_name, None)
        setattr(train_cfg, arg_name, value)

    """ Parameter Merging Approaches """
    if args.merge_or_route in ["uniform", "ties"]:
        if args.merge_or_route == "uniform":
            expert = WeightedLinearMerge(WeightedLinearMergeConfig()).transform(library)
        elif args.merge_or_route == "ties":
            cfg = TiesMergeConfig(top_k=args.transform_sparsity)
            expert = TiesMerge(cfg).transform(library)

        module = MultiExpertModel(**vars(expert.training_config)).to("cuda")
        module.add_expert_instance(expert, is_default=True)
    elif args.merge_or_route == "uniform_lora_after_op":
        # Here we merge the LoRA experts after the outer product we cannot really do it
        # with the lib transform, cause this would require storing large matrices in memory
        # Instead we do it with a uniform selector
        assert type(an_expert.expert_info.expert_config) == LoRAConfig
        train_cfg.router_selector = "uniform"
        train_cfg.lora_merge_after = True
        module = MultiExpertModel(**vars(train_cfg)).to("cuda")
        module.add_experts_from_library(library)
    elif args.merge_or_route == "base":
        module = ExpertModel(**vars(train_cfg))

    elif args.merge_or_route in ["phatgoose", "arrow", "avg_act"]:
        """Routing Approaches"""
        args.router_selector = f"{args.merge_or_route}_router"

        selector_config = SelectorConfig.from_training_config(args)

        module = MultiExpertModel(
            **vars(train_cfg), selector_config=selector_config
        ).to("cuda")

        module.add_experts_from_library(library)
        patch_prototypes(module, library, args)

    elif args.merge_or_route == "oracle":
        """TaskNameSelector"""
        args.router_selector = "task_selector"
        selector_config = SelectorConfig.from_training_config(args)

        module = MultiExpertModel(
            **vars(train_cfg), selector_config=selector_config
        ).to("cuda")
        module.add_experts_from_library(library)
    else:
        raise ValueError(f"Unknown merge_or_route {args.merge_or_route}")

    metric_logger = Selector.metric_logger

    if wandb.run is None and os.environ.get("WANDB_API_KEY"):
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "0shot_routing"),
            config=dict(module.hparams),
            name=os.environ.get("AMLT_JOB_NAME", None),
        )
        # update config
        wandb.config.update({f"cmd_args_{k}": v for k, v in vars(args).items()})

    if args.pipeline_eval_tasks is None:
        logger.info("`pipeline_eval_tasks` was not set, setting pipeline_eval_tasks='all'...")
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
        scores = eval_in_distribution(module, train_cfg, tasks)
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
        rouge = rouge_evaluator.evaluate(module, split="test", verbose=False)
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
    args = ExpertConfig.parse()
    run_eval(args)
