import copy
import logging
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from nqa_datamodule import NQADatamodule
from train_qa import KEArguments

from mttl.arguments import MultiExpertConfig
from mttl.logging import setup_logging
from mttl.models.base_model import BaseExpertModel, BaseExpertModelConfig
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelConfig,
    MoEModel,
    MoEModelConfig,
)
from mttl.models.hf.trainer import LMTrainer
from mttl.models.km_model import KMMoEModel, KMMoEModelConfig
from mttl.models.library.expert import Expert, ExpertInfo, load_expert
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.lightning.expert_module import ExpertModule, MoEModule
from mttl.utils import create_library, remote_login, upload_library

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class QAEvalArguments(KEArguments):
    ke_expert_name: str = "KE"


def eval_qa(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    # If we are loading Knowledge Modules, let's make sure to hardcode routing args
    if training_args.router_granularity != "coarsegrained":
        logger.warning("Overwriting `router_granularity` to 'coarsegrained'")
        training_args.router_granularity = "coarsegrained"

    if training_args.router_selector != "ke_selector":
        logger.warning("Overwriting `router_selector` to 'ke_selector'")
        training_args.router_selector = "ke_selector"

    # Build model (will have 0 experts if `library_id` is None)
    model_config = MoEModelConfig(
        base_model=training_args.model,
        library_id=None,
        selector_config=training_args.selector_config,
    )
    model = MoEModel(model_config)

    # Build evaluator, and fetch tasks names for which we need a KM
    training_args.dataset = training_args.nqa_dataset

    from nqa_evaluator import NQAZeroShotEvaluator

    evaluator = NQAZeroShotEvaluator(training_args, generation_kwargs={})

    if training_args.library_id:
        test_tasks = set(evaluator.datamodule.test_dataset["document_id"])
        expert_lib = ExpertLibrary.get_expert_library(
            training_args.library_id, selection=list(test_tasks)
        )
        model.add_experts_from_library(expert_lib)

    # We first try to load a QA expert if provided
    if training_args.ke_hf_path:
        ke_expert = load_expert(training_args.ke_hf_path)
        model.add_expert_instance(ke_expert, expert_name=training_args.ke_expert_name)

    model = model.cuda()

    # Call the NQA callback
    rougeL, predictions = evaluator.evaluate(
        model, split="test", return_predictions=True
    )

    print(f"ROUGE-L: {rougeL}")


if __name__ == "__main__":
    args = QAEvalArguments.parse()
    assert args.dataset_config
    eval_qa(args)
