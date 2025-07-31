# here, we train experts and we upload them to a local library (repository) of experts.

from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.evaluators.math500_evaluator import Math500Evaluator
from mttl.arguments import EvaluationConfig
from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
import torch
from mttl.logging import setup_logging

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_logging()

args = EvaluationConfig.parse()

from mttl.datamodule.math500_module import Math500DataConfig, Math500DataModule

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

config = Math500DataConfig(
    model=args.model,
    model_family=args.model_family,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    data_dir=args.output_dir
)

dm = Math500DataModule(config, for_generation=True)
evaluator = Math500Evaluator(dm)

if args.library_id is None:
    module = ExpertModule(**vars(args)).to(device)
else:
    library = ExpertLibrary.get_expert_library(args.library_id)
    module = MultiExpertModule(**vars(args)).to(device)

    if args.merge_or_route == "uniform":
        module.add_experts_from_library(args.library_id)
        module.model.set_selector("lora", UniformSelectorConfig(lora_merge_after=args.lora_merge_after))
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

# evaluate
result = evaluator.evaluate(module.model, split="test")
