import os
import sys
import json
import torch
import copy
from copy import deepcopy
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from mttl.models.modifiers.expert_containers.expert_library import get_expert_library
from mttl.models.modifiers.expert_containers.selectors import ClownSelector
from mttl.models.modifiers.lora import LoRAConfig


from pytorch_lightning import seed_everything
import json

from mttl.utils import logger, remote_login, setup_logging

from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    RoutedMultiExpertModel,
)
from projects.wiki_experts.src.config import ExpertConfig

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.models.modifiers.expert_containers.library_transforms import (
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
    HiddenStateComputer,
    HiddenStateComputerConfig,
    TiesMerge,
    TiesMergeConfig,
    SVDInputExtractor,
    SVDInputExtractorConfig,
)
from mttl.models.modifiers.expert_containers.expert_containers import ExpertContainer


def get_hidden_states(library, args):
    if args.delta_scale:
        cfg = HiddenStateComputerConfig(
            max_samples_per_task=args.max_samples_per_task,
            track=args.track,
            pool=args.pool,
            use_base_model_only=True,
        )
        base = HiddenStateComputer(cfg).transform(library, default_args=args)
        cfg.use_base_model_only = False
        expert = HiddenStateComputer(cfg).transform(library, default_args=args)
        output = {
            exp_name: {
                k: (expert[exp_name][k] - base[exp_name][k]) * args.delta_scale
                + base[exp_name][k]
                for k in base[exp_name].keys()
            }
            for exp_name in base.keys()
        }
    else:
        cfg = HiddenStateComputerConfig(
            use_base_model_only=args.use_base_model_only,
            max_samples_per_task=args.max_samples_per_task,
            track=args.track,
            pool=args.pool,
        )
        output = HiddenStateComputer(cfg).transform(library)

    return output


def get_svd_embeddings(library, args):
    cfg = SVDInputExtractorConfig(scale=args.scale_prototypes)
    svd_input_extractor = SVDInputExtractor(cfg)
    return svd_input_extractor.transform(library)


def patch_prototypes(module, library, args, proto_inits=None):
    if not proto_inits and args.proto_init == "svd":
        proto_inits = get_svd_embeddings(library, args)
    elif not proto_inits and args.proto_init == "hidden":
        proto_inits = get_hidden_states(library, args)

    for mod in module.modules():
        if isinstance(mod, ClownSelector):
            prototypes = []
            params = []
            for expert_name in mod.expert_names:
                layer_names = proto_inits[expert_name].keys()
                valid_layer_names = [
                    k for k in layer_names if k.startswith(mod.layer_name)
                ]
                key = sorted(valid_layer_names)[0]
                prototypes += [proto_inits[expert_name][key]]

            logger.info(
                f"setting prototypes for selector at {mod.layer_name} with hidden states from {key}"
            )
            prototypes = torch.stack(prototypes)
            if args.scale_prototypes:
                # prototypes are not normalized, we will at least make their norm smaller than 1
                max_norm = torch.norm(prototypes, dim=1, p=2).max()
                prototypes = prototypes / max_norm

            mod.overwrite_prototypes(prototypes)


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

    library = get_expert_library(
        repo_id=args.library_id,
        exclude_selection=exclude_phi_tasks,
    )

    if args.merge_or_route in ["uniform", "weighted"]:
        weights = None
        if args.merge_or_route == "weighted":
            # get weights from 10 cluster
            from mttl.datamodule import task_cluster_flan

            tasks = []
            for i in range(10):
                tasks += [getattr(task_cluster_flan, f"c{i}_2e")]
            weights = {}
            for task_subset in tasks:
                for task in task_subset:
                    weights[task] = 1 / len(task_subset) / 10

        expert = WeightedLinearMerge(
            WeightedLinearMergeConfig(weights=weights)
        ).transform(library)
        module = MultiExpertModel(**vars(expert.training_config)).to("cuda")
        module.add_expert_instance(expert, is_default=True)
    elif args.merge_or_route == "uniform_lora_after_op":
        # Here we merge the LoRa experts after the outer product
        # we cannot really do it with the lib transform, cause this would require storing large matrices in memory
        # Instead we do it with a uniform selector
        expert_names = list(library.keys())
        expert = copy.deepcopy(library[expert_names[0]])
        assert type(expert.expert_info.expert_config) == LoRAConfig
        config = expert.training_config
        config.router_selector = "uniform"
        config.lora_merge_after = True
        module = MultiExpertModel(**vars(config)).to("cuda")
        module.add_experts_from_library(library)
    elif args.merge_or_route == "ties":
        cfg = TiesMergeConfig(top_k=args.transform_sparsity)
        ties_expert = TiesMerge(cfg).transform(library)
        module = MultiExpertModel(**vars(ties_expert.training_config)).to("cuda")
        module.add_expert_instance(ties_expert, is_default=True)
    elif args.merge_or_route in ["clown", "clown_lora_after_op"]:
        an_expert = library[next(iter(library.keys()))]
        args_copy = deepcopy(an_expert.training_config)
        args_copy.router_selector = "clown_router"
        args_copy.router_temp = args.router_temp
        args_copy.moe_top_k = args.moe_top_k
        args_copy.precision = 32
        args_copy.router_window_size = args.router_window_size
        args_copy.clown_mode = args.clown_mode
        args_copy.proto_init = args.proto_init
        args_copy.normalize_router_input = args.normalize_router_input
        args_copy.lora_merge_after = args.merge_or_route == "clown_lora_after_op"
        module = RoutedMultiExpertModel(**vars(args_copy), device_map="auto")
        module.load_from_module_dict(library)
        patch_prototypes(module, library, args)
        module = module.to("cuda")
    elif args.merge_or_route == "phatgoose":
        from mttl.models.modifiers.expert_containers.library_transforms import (
            PhatgooseTransform,
            PhatgooseConfig,
        )

        an_expert = library[next(iter(library.keys()))]
        args_copy = deepcopy(an_expert.training_config)
        # phatgoose does merging after by default
        args_copy.lora_merge_after = True
        args_copy.router_selector = "phatgoose_selector"
        args_copy.router_temp = args.router_temp
        args_copy.moe_top_k = args.moe_top_k
        module = RoutedMultiExpertModel(**vars(args_copy), device_map="auto")
        module.load_from_module_dict(library)

        phagoose_transform = PhatgooseTransform(PhatgooseConfig(recompute=False))
        prototypes = phagoose_transform.transform(
            library, expert_names=None, default_args=args
        )
        # load prototypes into the router
        for mod in module.modules():
            if isinstance(mod, ExpertContainer):
                mod.selector.set_prototypes(prototypes)
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

    # try to fetch routing statistics
    routing_stats = {}
    if hasattr(module.model, "task_id_container"):
        for task_name in module.model.task_id_container.keys():
            if task_name == "routing_infos":
                continue

            task_dict = module.model.task_id_container[task_name]
            for k, v in task_dict.items():
                routing_stats[f"{task_name}/{k}"] = v

    if os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "0shot_routing"),
            config=dict(module.hparams),
            name=os.environ.get("AMLT_JOB_NAME", None),
        )
        wandb.log({f"downstream/{k}": v for k, v in scores.items()})

        if len(routing_stats) > 0:
            wandb.log(routing_stats)

        wandb.finish()


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
