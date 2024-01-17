import os
import sys
import json
from copy import deepcopy

from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from huggingface_hub import login
from pytorch_lightning import seed_everything
import json

from mttl.utils import logger, setup_logging

from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
)
from projects.wiki_experts.src.config import ExpertConfig

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.models.modifiers.expert_containers.library_transforms import (
    WeightedLinearMerge,
    WeightedLinearMergeConfig,
    DatasetCentroidComputer,
    PrototypeComputerConfig,
    TiesMerge,
    TiesMergeConfig,
)


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

    # cfg = TiesMergeConfig(top_k=0.2)
    # ties_expert = TiesMerge(cfg).transform(library)
    ave_cfg = WeightedLinearMergeConfig()
    ave_expert = WeightedLinearMerge(ave_cfg).transform(library)
    module = MultiExpertModel(**vars(ave_expert.training_config)).to("cuda")
    module.add_expert_instance(ave_expert, is_default=True)

    if args.pipeline_eval_tasks == "all":
        args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

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
    runner.run(module)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_multitask(args)
