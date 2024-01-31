import os
import sys
import json
import torch
from copy import deepcopy
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from mttl.models.modifiers.expert_containers.expert_library import (
    HFExpertLibrary,
    LocalExpertLibrary,
)
from mttl.models.modifiers.expert_containers.module_graph import Expert
from mttl.models.modifiers.expert_containers.selectors import ClownSelector


from huggingface_hub import login
from pytorch_lightning import seed_everything
import json

from mttl.utils import logger, setup_logging

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
    ExpertProjector,
    ExpertProjectorConfig,
    SVDEmbeddingTransform,
    SVDEmbeddingTransformConfig,
    SVDInputExtractor,
    SVDInputExtractorConfig,
)


def get_hidden_states(library, args):
    if "phi" in library.repo_id:
        centroid_name = "dataset_centroids"
    else:
        centroid_name = "dataset_centroids_2"

    if args.delta_scale:
        cfg = HiddenStateComputerConfig(
            name=centroid_name,
            max_samples_per_task=args.max_samples_per_task,
            track=args.track,
            pool=args.pool,
            upload_to_hf=True,
            # compute_delta=False,
            use_base_model_only=True,
        )
        base = HiddenStateComputer(cfg).transform(library, default_args=args)
        # cfg.compute_delta = True
        # delta = HiddenStateComputer(cfg).transform(library, default_args=args)
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
            name=centroid_name,
            use_base_model_only=args.use_base_model_only,
            max_samples_per_task=args.max_samples_per_task,
            track=args.track,
            pool=args.pool,
            upload_to_hf=True,
        )
        output = HiddenStateComputer(cfg).transform(library)

    return output


def get_svd_embeddings(library, args):
    cfg = SVDInputExtractorConfig(
        top_k=args.transform_sparsity, recompute=False, scale=args.scale_prototypes
    )
    svd_input_extractor = SVDInputExtractor(cfg)
    return svd_input_extractor.transform(library)


def patch_prototypes(module, library, args):
    if args.proto_init == "svd":
        proto_inits = get_svd_embeddings(library, args)
    elif args.proto_init == "hidden":
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

                if args.use_similarity_scaling:
                    # get params too
                    expert = library[expert_name]
                    expert_params = [
                        expert.expert_weights[k] for k in valid_layer_names
                    ]
                    expert_params = torch.cat([p.flatten() for p in expert_params])
                    params += [expert_params]

            logger.info(
                f"setting prototypes for selector at {mod.layer_name} with hidden states from {key}"
            )
            prototypes = torch.stack(prototypes)
            if args.scale_prototypes:
                # prototypes are not normalized, we will at least make their norm smaller than 1
                max_norm = torch.norm(prototypes, dim=1, p=2).max()
                prototypes = prototypes / max_norm

            if args.use_similarity_scaling:
                params = torch.stack(params)
                sim = F.cosine_similarity(
                    params.unsqueeze(0), params.unsqueeze(1), dim=-1
                )
                sim = (sim - sim.min()) / (sim.max() - sim.min())
                prototypes = torch.einsum(
                    "BC,CD->BD", sim.type_as(prototypes), prototypes
                )

            mod.overwrite_prototypes(prototypes)


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    exclude_phi_tasks = [
        "hellaswag_1_1_0",
        "ai2_arc_ARC_Challenge_1_0_0",
        "ai2_arc_ARC_Easy_1_0_0",
        "piqa_1_0_0",
        "winogrande_1_1_0",
        "bool_q_1_0_0",
        "openbookqa_0_1_0",
    ]

    library = HFExpertLibrary(
        repo_id=args.hf_lib_id, exclude_selection=exclude_phi_tasks
    )

    if args.merge_or_route in ["uniform", "weighted"]:
        weights = None
        if args.merge_or_route == "weighted":
            # get weights from 10 cluster
            from mttl.datamodule import task_cluster_flan

            tasks = []
            for i in range(10):
                tasks += [getattr(task_cluster_flan, f"c{i}_2e")]
            n_tasks = sum([len(t) for t in tasks])
            weights = {}
            for task_subset in tasks:
                for task in task_subset:
                    weights[task] = 1 / len(task_subset) / 10

        expert = WeightedLinearMerge(
            WeightedLinearMergeConfig(weights=weights)
        ).transform(library)
        module = MultiExpertModel(**vars(expert.training_config)).to("cuda")
        module.add_expert_instance(expert, is_default=True)
    elif args.merge_or_route == "ties":
        cfg = TiesMergeConfig(top_k=args.transform_sparsity)
        ties_expert = TiesMerge(cfg).transform(library)
        module = MultiExpertModel(**vars(ties_expert.training_config)).to("cuda")
        module.add_expert_instance(ties_expert, is_default=True)
    elif args.merge_or_route == "clown":
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

        module = RoutedMultiExpertModel(**vars(args_copy), device_map="auto")
        module.load_from_module_dict(library)
        patch_prototypes(module, library, args)
        module = module.to("cuda")

    if args.pipeline_eval_tasks == "all":
        args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

    with torch.no_grad():
        runner: EvaluatorRunner = setup_evaluators(
            model_type=module.hparams.model,
            model_family=module.hparams.model_family,
            max_input_length=module.hparams.max_input_length,
            max_output_length=module.hparams.max_output_length,
            predict_batch_size=4,
            truncation_side=module.hparams.truncation_side,
            tasks=args.pipeline_eval_tasks,
            output_path=os.path.join(args.output_dir, "DOWNSTREAM"),
        )
        scores = runner.run(module)

    if os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb.init(project="0shot_routing", config=args.asdict())
        wandb.log({f"downstream/{k}": v for k, v in scores.items()})
        wandb.finish()


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
