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
device = "cuda" if torch.cuda.is_available() else "cpu"
setup_logging()

args = EvaluationConfig.parse()


def consine_similarity_principal_components(delta_code, delta_math, layer):
    U_code, _, _ = torch.linalg.svd(delta_code, full_matrices=False)
    U_math, _, _ = torch.linalg.svd(delta_math, full_matrices=False)

    # Cosine of top-1 singular vector
    top1_cosine = torch.abs(torch.dot(U_math[:, 0], U_code[:, 0]))
    print(f"Top-1 singular vector cosine similarity in the {layer} layer: {top1_cosine:.4f}")
    return top1_cosine

def subspace_preservation(delta_code, delta_math, layer):
    U_code, _, _ = torch.linalg.svd(delta_code, full_matrices=False)
    U_math, _, _ = torch.linalg.svd(delta_math, full_matrices=False)
    rank = 1
    U_code_k = U_code[:, :rank]
    U_math_k = U_math[:, :rank]
    
    overlap_score = torch.norm(U_math_k.T @ U_code_k, p='fro')
    print(f"Subspace preservation score in the {layer} layer: {overlap_score:.4f}")
    return overlap_score

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
    print(f"Effective rank - math: {rank_math}, code: {rank_code}, merged: {rank_merge} in the {layer} layer")
    return rank_math, rank_code, rank_merge

def spectral_energy_increase(delta_code, delta_math, layer):
    delta_merge = delta_math + delta_code
    energy_math = torch.norm(delta_math, p='fro')**2
    energy_code = torch.norm(delta_code, p='fro')**2
    energy_merge = torch.norm(delta_merge, p='fro')**2

    print(f"Spectral energy - math: {energy_math:.4f}, code: {energy_code:.4f}, merge: {energy_merge:.4f} in the {layer} layer")
    print(f"Sum of individual energies: {(energy_math + energy_code):.4f} in the {layer} layer")
    return energy_math, energy_code, energy_merge

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

        for layer in range(len(module.model.model.model.layers)):
            expert_weights_q = []
            expert_weights_k = []
            expert_weights_v = []
            expert_weights_up = []
            expert_weights_down = []
            for expert in experts:
                q_proj_lora_a = module.model.model.model.layers[
                    layer
                ].self_attn.q_proj.lora_a[expert]
                q_proj_lora_b = module.model.model.model.layers[
                    layer
                ].self_attn.q_proj.lora_b[expert]

                k_proj_lora_a = module.model.model.model.layers[
                    layer
                ].self_attn.k_proj.lora_a[expert]
                k_proj_lora_b = module.model.model.model.layers[
                    layer
                ].self_attn.k_proj.lora_b[expert]

                v_proj_lora_a = module.model.model.model.layers[
                    layer
                ].self_attn.v_proj.lora_a[expert]
                v_proj_lora_b = module.model.model.model.layers[
                    layer
                ].self_attn.v_proj.lora_b[expert]

                up_proj_lora_a = module.model.model.model.layers[layer].mlp.up_proj.lora_a[expert]
                up_proj_lora_b = module.model.model.model.layers[layer].mlp.up_proj.lora_b[expert]

                down_proj_lora_a = module.model.model.model.layers[
                    layer
                ].mlp.down_proj.lora_a[expert]
                down_proj_lora_b = module.model.model.model.layers[
                    layer
                ].mlp.down_proj.lora_b[expert]

                expert_weight_q = q_proj_lora_a @ q_proj_lora_b
                expert_weight_k = k_proj_lora_a @ k_proj_lora_b
                expert_weight_v = v_proj_lora_a @ v_proj_lora_b
                expert_weight_up = up_proj_lora_a @ up_proj_lora_b
                expert_weight_down = down_proj_lora_a @ down_proj_lora_b

                expert_weights_q.append(expert_weight_q)
                expert_weights_k.append(expert_weight_k)
                expert_weights_v.append(expert_weight_v)
                expert_weights_up.append(expert_weight_up)
                expert_weights_down.append(expert_weight_down)

            cos_q = consine_similarity_principal_components(expert_weights_q[0].cpu().detach(), expert_weights_q[1].cpu().detach(), layer)
            cos_k = consine_similarity_principal_components(expert_weights_k[0].cpu().detach(), expert_weights_k[1].cpu().detach(), layer)
            cos_v = consine_similarity_principal_components(expert_weights_v[0].cpu().detach(), expert_weights_v[1].cpu().detach(), layer)
            cos_up = consine_similarity_principal_components(expert_weights_up[0].cpu().detach(), expert_weights_up[1].cpu().detach(), layer)
            cos_down = consine_similarity_principal_components(expert_weights_down[0].cpu().detach(), expert_weights_down[1].cpu().detach(), layer)
            
            subspace_preservation_q = subspace_preservation(expert_weights_q[0].cpu().detach(), expert_weights_q[1].cpu().detach(), layer)
            subspace_preservation_k = subspace_preservation(expert_weights_k[0].cpu().detach(), expert_weights_k[1].cpu().detach(), layer)
            subspace_preservation_v = subspace_preservation(expert_weights_v[0].cpu().detach(), expert_weights_v[1].cpu().detach(), layer)
            subspace_preservation_up = subspace_preservation(expert_weights_up[0].cpu().detach(), expert_weights_up[1].cpu().detach(), layer)
            subspace_preservation_down = subspace_preservation(expert_weights_down[0].cpu().detach(), expert_weights_down[1].cpu().detach(), layer)

            rank_math_q, rank_code_q, rank_merge_q = effective_rank_analysis(expert_weights_q[0].cpu().detach(), expert_weights_q[1].cpu().detach(), layer)
            rank_math_k, rank_code_k, rank_merge_k = effective_rank_analysis(expert_weights_k[0].cpu().detach(), expert_weights_k[1].cpu().detach(), layer)
            rank_math_v, rank_code_v, rank_merge_v = effective_rank_analysis(expert_weights_v[0].cpu().detach(), expert_weights_v[1].cpu().detach(), layer)
            rank_math_up, rank_code_up, rank_merge_up = effective_rank_analysis(expert_weights_up[0].cpu().detach(), expert_weights_up[1].cpu().detach(), layer)
            rank_math_down, rank_code_down, rank_merge_down = effective_rank_analysis(expert_weights_down[0].cpu().detach(), expert_weights_down[1].cpu().detach(), layer)
            energy_math_q, energy_code_q, energy_merge_q = spectral_energy_increase(expert_weights_q[0].cpu().detach(), expert_weights_q[1].cpu().detach(), layer)
            energy_math_k, energy_code_k, energy_merge_k = spectral_energy_increase(expert_weights_k[0].cpu().detach(), expert_weights_k[1].cpu().detach(), layer)
            energy_math_v, energy_code_v, energy_merge_v = spectral_energy_increase(expert_weights_v[0].cpu().detach(), expert_weights_v[1].cpu().detach(), layer)
            energy_math_up, energy_code_up, energy_merge_up = spectral_energy_increase(expert_weights_up[0].cpu().detach(), expert_weights_up[1].cpu().detach(), layer)
            energy_math_down, energy_code_down, energy_merge_down = spectral_energy_increase(expert_weights_down[0].cpu().detach(), expert_weights_down[1].cpu().detach(), layer)

            write_json = {
                f"cos_q_{layer}": cos_q,
                f"cos_k_{layer}": cos_k,
                f"cos_v_{layer}": cos_v,
                f"cos_up_{layer}": cos_up,
                f"cos_down_{layer}": cos_down,
                f"subspace_preservation_q_{layer}": subspace_preservation_q,
                f"subspace_preservation_k_{layer}": subspace_preservation_k,
                f"subspace_preservation_v_{layer}": subspace_preservation_v,
                f"subspace_preservation_up_{layer}": subspace_preservation_up,
                f"subspace_preservation_down_{layer}": subspace_preservation_down,
                f"rank_math_q_{layer}": rank_math_q,
                f"rank_code_q_{layer}": rank_code_q,
                f"rank_merge_q_{layer}": rank_merge_q,
                f"rank_math_k_{layer}": rank_math_k,
                f"rank_code_k_{layer}": rank_code_k,
                f"rank_merge_k_{layer}": rank_merge_k,
                f"rank_math_v_{layer}": rank_math_v,
                f"rank_code_v_{layer}": rank_code_v,
                f"rank_merge_v_{layer}": rank_merge_v,
                f"rank_math_up_{layer}": rank_math_up,
                f"rank_code_up_{layer}": rank_code_up,
                f"rank_merge_up_{layer}": rank_merge_up,
                f"rank_math_down_{layer}": rank_math_down,
                f"rank_code_down_{layer}": rank_code_down,
                f"rank_merge_down_{layer}": rank_merge_down,
                f"energy_math_q_{layer}": energy_math_q,
                f"energy_code_q_{layer}": energy_code_q,
                f"energy_merge_q_{layer}": energy_merge_q,
                f"energy_math_k_{layer}": energy_math_k,
                f"energy_code_k_{layer}": energy_code_k,
                f"energy_merge_k_{layer}": energy_merge_k,
                f"energy_math_v_{layer}": energy_math_v,
                f"energy_code_v_{layer}": energy_code_v,
                f"energy_merge_v_{layer}": energy_merge_v,
                f"energy_math_up_{layer}": energy_math_up,
                f"energy_code_up_{layer}": energy_code_up,
                f"energy_merge_up_{layer}": energy_merge_up,
                f"energy_math_down_{layer}": energy_math_down,
                f"energy_code_down_{layer}": energy_code_down,
                f"energy_merge_down_{layer}": energy_merge_down,
            }
            file.write(json.dumps(write_json) + "\n")
            file.flush()
            

        # def print_spectral_metrics(expert_weights, original_weights, layer, expert):
        #     # Convert tensors to same dtype and device
        #     original_weights = original_weights.to(expert_weights.dtype).to(
        #         expert_weights.device
        #     )

        #     # Detach tensors for computation
        #     expert_weights = expert_weights.detach()
        #     original_weights = original_weights.detach()

        #     # Calculate and print metrics
        #     dist = spectral_distance(expert_weights, original_weights)
        #     ratio = spectral_energy_ratio(expert_weights, original_weights)

        #     print(f"Layer {layer} Expert {expert} Spectral Distance: {dist}")
        #     print(f"Layer {layer} Expert {expert} Spectral Energy Ratio: {ratio}")

        #     return dist, ratio

        # # Calculate expert weights and call print function
        # expert_weights = q_proj_lora_a @ q_proj_lora_b
        # original_weights = module.model.model.model.layers[
        #     layer
        # ].self_attn.q_proj.layer.weight
        # print_spectral_metrics(expert_weights, original_weights, layer, expert)

        #     

        #     

        #     print(q_proj_lora_a.shape)
        #     print(q_proj_lora_b.shape)

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
