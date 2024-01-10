import os
import sys
import json
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from huggingface_hub import login
from pytorch_lightning import seed_everything
import json

from mttl.utils import logger, setup_logging

from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    MoETrainer,
    ExpertTrainer,
)
from projects.wiki_experts.src.config import ExpertConfig
from mttl.models.modifiers.expert_containers.library_transforms import WeightedExpert


def update_modifier_args_from_ckpt(args, ckpt_args):
    from projects.wiki_experts.src.config import ExpertConfig

    ckpt_training_config = ExpertConfig.fromdict(ckpt_args)

    for name, value in vars(ckpt_training_config).items():
        current_value = getattr(args, name, None)
        if (current_value is None and value is not None) or current_value != value:
            logger.info(f"overwriting {name} from {current_value} to {value}")
            setattr(args, name, value)

    args.model_name = ckpt_args["model"]


def run_multitask(args: ExpertConfig):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    transform = WeightedExpert(args)
    # get the config of an expert in the library
    library = transform.library
    exp = library[next(iter(library.keys()))]
    tr_cfg, exp_cfg = exp.training_config, exp.expert_config

    # create a base model, according to the loaded training config
    model_args = deepcopy(args)
    update_modifier_args_from_ckpt(model_args, vars(tr_cfg))

    # create backbone
    # module = MultiExpertModel(**vars(model_args))
    # module = MoETrainer(**vars(model_args))
    module = ExpertTrainer(**vars(model_args))
    breakpoint()
    # create weighted expert
    # weighted_expert = transform.compute(return_expert=True)
    # add weighted expert
    # module.add_expert_instance(exp, is_default=True)

    from mttl.evaluators.base import EvaluatorRunner, setup_evaluators

    runner: EvaluatorRunner = setup_evaluators(
        model_type=model_args.model,
        model_family=model_args.model_family,
        max_input_length=model_args.max_input_length,
        max_output_length=model_args.max_output_length,
        predict_batch_size=args.predict_batch_size,
        truncation_side=model_args.truncation_side,
        tasks=args.pipeline_eval_tasks,
        output_path=os.path.join(args.output_dir, "DOWNSTREAM"),
    )

    metrics = runner.run(module)
    breakpoint()


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
