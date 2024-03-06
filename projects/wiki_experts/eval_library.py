import os
import sys
import torch
import copy
import wandb
import numpy as np
from copy import deepcopy
from pytorch_lightning import seed_everything
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.expert_containers.selectors import PerTokenSelector
from mttl.models.modifiers.lora import LoRAConfig

from mttl.utils import logger, remote_login, setup_logging
from mttl.models.expert_model import MultiExpertModel
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

from mttl.models.modifiers.expert_containers.expert_containers import ExpertContainer
from mttl.callbacks import LossCallback
from mttl.datamodule.base import get_datamodule
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from projects.wiki_experts.src.utils.utils import TableLogger


def get_hidden_states(library, args):
    cfg = HiddenStateComputerConfig(
        use_base_model_only=args.use_base_model_only,
        max_samples_per_task=args.max_samples_per_task,
        name=args.expert_embeds_save_name,
        track=args.track,
        pool=args.pool,
    )
    output = HiddenStateComputer(cfg).transform(
        library, recompute=args.recompute_prototypes
    )

    return output


def get_arrow_embeddings(library, args):
    cfg = ArrowConfig(
        name=args.expert_embeds_save_name,
    )
    return ArrowTransform(cfg).transform(library, recompute=args.recompute_prototypes)


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
    if not proto_inits and args.proto_init == "arrow":
        proto_inits = get_arrow_embeddings(library, args)
    elif not proto_inits and args.proto_init == "hidden":
        proto_inits = get_hidden_states(library, args)
    elif not proto_inits and args.proto_init == "phatgoose":
        proto_inits = get_phatgoose_embeddings(library, args)

    for mod in module.modules():
        if isinstance(mod, PerTokenSelector):
            patched_layer_name = mod.layer_name.replace(".selector", "")
            prototypes = []
            for expert_name in mod.expert_names:
                patched_layer_name = mod.layer_name.replace(".selector", "")
                layer_names = proto_inits[expert_name].keys()
                patched_layer_name = mod.layer_name.replace(".selector", "")
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


def eval_in_distribution(module, args: ExpertConfig, tasks):
    args.include_task_source = "*"
    transfer_table = TableLogger()

    for task in tasks:
        args.finetune_task_name = task
        args.predict_batch_size = 16
        if args.eval_metric in ["val_loss", "loss"]:
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.val_dataloader(), output_dir=args.output_dir, name=task + "_val"
            )
            metric = evaluator.test(pl_module=module)

        elif args.eval_metric == "test_loss":
            dm = get_datamodule(args)
            evaluator = LossCallback(
                dm.test_dataloader(), output_dir=args.output_dir, name=task + "_test"
            )
            metric = evaluator.test(model=module)
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


def run_multitask(args: ExpertConfig):
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
        exclude_selection=exclude_phi_tasks,
        # make_local=True,
    )
    an_expert = library[next(iter(library.keys()))]
    train_cfg = deepcopy(an_expert.training_config)

    if args.merge_or_route in ["uniform", "weighted"]:
        weights = None
        if args.merge_or_route == "weighted":
            # get weights from 10 cluster
            _task_cluster_flan = json.load(
                open(
                    "projects/wiki_experts/task_sets/phi/clusters_kmeans_simmatrix_phi2_v3.json"
                )
            )

            tasks = []
            for i in range(10):
                tasks += _task_cluster_flan[f"c{i}_2e"]
            weights = {}
            for task_subset in tasks:
                for task in task_subset:
                    weights[task] = 1 / len(task_subset) / 10

        expert = WeightedLinearMerge(
            WeightedLinearMergeConfig(weights=weights)
        ).transform(library)

        module = MultiExpertModel(**vars(expert.training_config))
        module.add_expert_instance(expert, is_default=True)
        train_cfg.output_dir = args.output_dir
        train_cfg.eval_metric = args.eval_metric
    elif args.merge_or_route == "uniform_lora_after_op":
        # Here we merge the LoRA experts after the outer product
        # we cannot really do it with the lib transform, cause this would require storing large matrices in memory
        # Instead we do it with a uniform selector
        assert type(expert.expert_info.expert_config) == LoRAConfig
        train_cfg.router_selector = "uniform"
        train_cfg.lora_merge_after = True
        module = MultiExpertModel(**vars(train_cfg))
        module.add_experts_from_library(library)
    elif args.merge_or_route == "ties":
        cfg = TiesMergeConfig(top_k=args.transform_sparsity)
        ties_expert = TiesMerge(cfg).transform(library)
        module = MultiExpertModel(**vars(ties_expert.training_config))
        module.add_expert_instance(ties_expert, is_default=True)
    elif args.merge_or_route == "route":
        train_cfg.router_selector = "per_token_router"
        train_cfg.precision = 32
        train_cfg.proto_init = args.proto_init

        # Proto init
        if args.proto_init == "arrow":
            # Set default arrow params, unless they are overwritten
            train_cfg.router_temp = args.router_temp or 1
            train_cfg.moe_top_k = args.moe_top_k or -1
            train_cfg.input_norm_fn = args.input_norm_fn  # or None
            train_cfg.proto_norm_fn = args.proto_norm_fn  # or None
        elif args.proto_init == "phatgoose":
            # TODO: overwrite to default values ?
            train_cfg.router_temp = args.router_temp or -1  # defaults to sqrt(d)
            train_cfg.moe_top_k = args.moe_top_k or 2
            train_cfg.input_norm_fn = args.input_norm_fn or "layer_norm"
            train_cfg.proto_norm_fn = args.proto_norm_fn or "layer_norm"
        else:
            assert args.proto_init == "hidden"

        module = MultiExpertModel(**vars(train_cfg), device_map="auto")
        module.load_from_module_dict(library)
        patch_prototypes(module, library, args)

    elif args.merge_or_route == "oracle":
        # TaskNameSelector
        train_cfg.output_dir = args.output_dir
        train_cfg.router_selector = "task_selector"
        module = MultiExpertModel(**vars(train_cfg), device_map="auto")
        module.load_from_module_dict(library)

    module = module.to("cuda")

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
        )
        scores = runner.run(module)

    if args.pipeline_eval_tasks == "in_distribution":
        tasks = [expert.expert_task_name for expert in library.data.values()]
        args_copy.eval_metric = args.eval_metric
        scores = eval_in_distribution(module, args_copy or args, tasks)
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
            )
            scores = runner.run(module)

        # try to fetch routing statistics
        routing_stats = {}
        if hasattr(module.model, "task_id_container"):
            for task_name in module.model.task_id_container.keys():
                if task_name == "routing_infos":
                    continue

            task_dict = module.model.task_id_container[task_name]
            for k, v in task_dict.items():
                routing_stats[f"{task_name}/{k}"] = v

    if wandb.run is not None:
        wandb.log({f"downstream/{k}": v for k, v in scores.items()})
        if len(routing_stats) > 0:
            wandb.log(routing_stats)

        wandb.finish()


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
