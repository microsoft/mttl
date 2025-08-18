# here, we train experts and we upload them to a local library (repository) of experts.

from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.evaluators.gsm_evaluator import GsmEvaluator
from mttl.arguments import EvaluationConfig
from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
import torch
from tqdm import tqdm
from mttl.logging import setup_logging
from mttl.models.modifiers.lora import spectral_distance, spectral_energy_ratio
import numpy as np
import matplotlib.pyplot as plt
import json
from mttl.models.library.library_transforms import WudiMerge, WudiMergeConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_logging()

args = EvaluationConfig.parse()


def consine_similarity_principal_components(delta_code, delta_math, layer):
    U_code, _, _ = torch.linalg.svd(delta_code, full_matrices=False)
    U_math, _, _ = torch.linalg.svd(delta_math, full_matrices=False)

    # Cosine of top-1 singular vector
    top1_cosine = torch.abs(torch.dot(U_math[:, 0], U_code[:, 0]))
    print(
        f"Top-1 singular vector cosine similarity in the {layer} layer: {top1_cosine:.4f}"
    )
    return top1_cosine.item()


def subspace_preservation(delta_code, delta_math, layer):
    U_code, _, _ = torch.linalg.svd(delta_code, full_matrices=False)
    U_math, _, _ = torch.linalg.svd(delta_math, full_matrices=False)
    rank = 1
    U_code_k = U_code[:, :rank]
    U_math_k = U_math[:, :rank]

    overlap_score = torch.norm(U_math_k.T @ U_code_k, p="fro")
    print(f"Subspace preservation score in the {layer} layer: {overlap_score:.4f}")
    return overlap_score.item()


def effective_rank_analysis(delta_code, delta_math, layer):
    U_code, _, _ = torch.linalg.svd(delta_code, full_matrices=False)
    U_math, _, _ = torch.linalg.svd(delta_math, full_matrices=False)

    delta_merge = delta_math + delta_code

    def effective_rank(s, threshold=0.01):
        return (s > threshold * s.max()).sum().item()

    # Compute singular values
    _, S_merge, _ = torch.linalg.svd(delta_merge, full_matrices=False)
    _, S_math, _ = torch.linalg.svd(delta_math, full_matrices=False)
    _, S_code, _ = torch.linalg.svd(delta_code, full_matrices=False)

    rank_math = effective_rank(S_math)
    rank_code = effective_rank(S_code)
    rank_merge = effective_rank(S_merge)
    print(
        f"Effective rank - math: {rank_math}, code: {rank_code}, merged: {rank_merge} in the {layer} layer"
    )
    return rank_math, rank_code, rank_merge


def spectral_energy_increase(delta_code, delta_math, layer):
    delta_merge = delta_math + delta_code
    energy_math = torch.norm(delta_math, p="fro") ** 2
    energy_code = torch.norm(delta_code, p="fro") ** 2
    energy_merge = torch.norm(delta_merge, p="fro") ** 2

    print(
        f"Spectral energy - math: {energy_math:.4f}, code: {energy_code:.4f}, merge: {energy_merge:.4f} in the {layer} layer"
    )
    print(
        f"Sum of individual energies: {(energy_math + energy_code):.4f} in the {layer} layer"
    )
    return energy_math.item(), energy_code.item(), energy_merge.item()


def compute_analysis(delta_code, delta_math, layer):
    # SVD for each matrix
    rank = 4

    # Fix the Code
    U_code, _, _ = torch.linalg.svd(delta_code, full_matrices=False)

    U_code_k = U_code[:, :rank]

    # Sweep over alpha values (a, with b = 1)
    alpha_values = np.linspace(0, 1.0, 10)
    preservation_scores = []
    top1_overlap_cos = []

    for a in tqdm(alpha_values):
        delta_ab = a * delta_math + delta_code
        U_ab, _, _ = torch.linalg.svd(delta_ab, full_matrices=False)
        U_ab_k = U_ab[:, :rank]

        # Frobenius norm of projection (subspace preservation)
        preservation = np.linalg.norm(U_code_k.T @ U_ab_k, ord="fro") / rank
        preservation_scores.append(preservation)

        # Cosine similarity of top-1 singular vectors
        top1_cos = np.abs(np.dot(U_code_k[:, 0], U_ab_k[:, 0]))
        top1_overlap_cos.append(top1_cos)
    # Plot results

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(
        alpha_values, preservation_scores, label="Subspace Preservation (Frobenius)"
    )
    plt.xlabel("Math LoRA Scaling Coefficient (a)")
    plt.ylabel("Preservation Score")
    plt.title("Code Subspace Preservation vs. Math LoRA Weight")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        alpha_values, top1_overlap_cos, label="Top-1 Cosine Similarity", color="orange"
    )
    plt.xlabel("Math LoRA Scaling Coefficient (a)")
    plt.ylabel("Cosine Similarity")
    plt.title("Top-1 Spectral Direction Similarity")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"analysis_{layer}.png")
    plt.show()


from mttl.datamodule.gsm_data_module import (
    GsmDataConfig,
    Gsm8kHardDataModule,
    Gsm8kDataModule,
    Gsm8kPerturbDataModule,
)
from mttl.models.library.library_transforms import (
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
)


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


config = GsmDataConfig(
    model=args.model,
    model_family=args.model_family,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    gsm_template=args.gsm_template,
    data_dir=args.output_dir,
)

if args.gsm_dataset == "gsm-hard":
    dm = Gsm8kHardDataModule(config, for_generation=True)
elif args.gsm_dataset == "gsm":
    dm = Gsm8kDataModule(config, for_generation=True)
elif args.gsm_dataset == "gsm-perturb":
    dm = Gsm8kPerturbDataModule(config, for_generation=True)

evaluator = GsmEvaluator(dm)

if args.library_id is None:
    module = ExpertModule(**vars(args)).to(device)
else:

    library = ExpertLibrary.get_expert_library(args.library_id)
    module = MultiExpertModule(**vars(args)).to(device)
    file = open("analysis.jsonl", "w")
    if args.merge_or_route == "uniform":
        module.add_experts_from_library(args.library_id)

        experts = library.keys()
        layer_count = len(module.model.model.model.layers)

        def get_expert_weights(layer, expert, proj_type):
            """Helper function to get expert weights for a projection type"""
            layer_obj = module.model.model.model.layers[layer]
            if proj_type in ["q", "k", "v"]:
                lora_a = getattr(layer_obj.self_attn, f"{proj_type}_proj").lora_a[
                    expert
                ]
                lora_b = getattr(layer_obj.self_attn, f"{proj_type}_proj").lora_b[
                    expert
                ]
            else:  # up or down
                lora_a = getattr(layer_obj.mlp, f"{proj_type}_proj").lora_a[expert]
                lora_b = getattr(layer_obj.mlp, f"{proj_type}_proj").lora_b[expert]
            return lora_a @ lora_b

        def compute_metrics(weights_list, layer, metric_func):
            """Helper function to compute metrics for a list of weights"""
            return metric_func(
                weights_list[0].cpu().detach(), weights_list[1].cpu().detach(), layer
            )

        # for layer in range(layer_count):
        #     # Collect expert weights for each projection type
        #     proj_types = ["q", "k", "v", "up", "down"]
        #     expert_weights = {pt: [] for pt in proj_types}

        #     for expert in experts:
        #         for pt in proj_types:
        #             expert_weights[pt].append(get_expert_weights(layer, expert, pt))

        #     # Compute metrics for each projection type
        #     metrics = {}
        #     for pt in proj_types:
        #         weights = expert_weights[pt]
        #         metrics.update(
        #             {
        #                 f"cos_{pt}_{layer}": compute_metrics(
        #                     weights, layer, consine_similarity_principal_components
        #                 ),
        #                 f"subspace_preservation_{pt}_{layer}": compute_metrics(
        #                     weights, layer, subspace_preservation
        #                 ),
        #             }
        #         )

        #         rank_metrics = compute_metrics(weights, layer, effective_rank_analysis)
        #         metrics.update(
        #             {
        #                 f"rank_math_{pt}_{layer}": rank_metrics[0],
        #                 f"rank_code_{pt}_{layer}": rank_metrics[1],
        #                 f"rank_merge_{pt}_{layer}": rank_metrics[2],
        #             }
        #         )

        #         energy_metrics = compute_metrics(
        #             weights, layer, spectral_energy_increase
        #         )
        #         metrics.update(
        #             {
        #                 f"energy_math_{pt}_{layer}": energy_metrics[0],
        #                 f"energy_code_{pt}_{layer}": energy_metrics[1],
        #                 f"energy_merge_{pt}_{layer}": energy_metrics[2],
        #             }
        #         )

        #     # Write results
        #     file.write(json.dumps(metrics) + "\n")
        #     file.flush()

        experts_names = library.keys()
        if args.expert_weights:
            module.model.set_selector(
                "lora",
                UniformSelectorConfig(
                    lora_merge_after=args.lora_merge_after,
                    experts_weight_list=args.expert_weights,
                ),
            )
        else:
            module.model.set_selector(
                "lora",
                UniformSelectorConfig(
                    lora_merge_after=args.lora_merge_after, experts_weight_list=None
                ),
            )
    elif args.merge_or_route == "wudi":
        cfg = WudiMergeConfig(iter=300, lr=1e-5)
        expert = WudiMerge(cfg).transform(library)

        module.model.add_expert_instance(expert, is_default=True)
    elif args.merge_or_route in ["phatgoose", "arrow", "avg_act"]:
        module.add_experts_from_library(args.library_id)
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
        module.model.set_selector("lora", selector_config)

    elif args.expert_selection is not None:
        expert = library.get_expert(args.expert_selection)
        module.add_expert_instance(expert, is_default=True)
    else:
        module = MultiExpertModule(**vars(args)).to(device)
        module.add_experts_from_library(args.library_id)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
    module.load_state_dict(checkpoint)


## evaluate
result = evaluator.evaluate(module.model, split="test")
