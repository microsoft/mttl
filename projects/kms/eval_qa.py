import copy
import json
import logging
import os
import time
from dataclasses import dataclass

from lightning_fabric import seed_everything

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule
from projects.kms.utils.nqa_datamodule import NQADatamodule

# isort: split
from mttl.arguments import ExpertConfig, MultiExpertConfig
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.logging import logger, setup_logging
from mttl.models.containers.selectors.km_selector import (
    KnowledgeExtractorSelectorConfig,
)
from mttl.models.expert_model import MoEModel
from mttl.models.km_model import KEMoEModelConfig
from mttl.models.library.expert import load_expert
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import remote_login
from projects.kms.train_km_simple import (
    evaluate_class,
    evaluate_datasets,
    evaluate_metrics,
)
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.arguments import MultiExpertConfig
from projects.kms.train_qa import KEArguments, train_datasets
from projects.kms.utils.nqa_evaluator import NQAZeroShotEvaluator
from projects.kms.utils.quality_evaluator import QualityEvaluator
from projects.kms.utils.wiki_mmlu_evaluator import WikiMMLUEvaluator


def load_task_splits(task_file):
    with open(task_file, "r") as f:
        split_dict = json.load(f)
    return split_dict


@dataclass
class QAEvalArguments(MultiExpertConfig):
    ke_expert_name: str = "KE"
    # split
    split: str = "test"
    # Where to load the KE expert from
    ke_hf_path: str = None
    # Which datamodule to use
    evaluate_on: str = None


def eval_qa(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    try:
        remote_login(training_args.remote_token)
    except Exception as e:
        logger.error(f"Failed to login remotely: {e}")

    # Build model (will have 0 experts if `library_id` is None)
    model_config = KEMoEModelConfig(
        base_model=training_args.model,
        library_id=None,
        expert_selection=training_args.finetune_task_name,
        selector_config=training_args.selector_config,
    )

    # create a model without any experts
    device = get_device()
    model = MoEModel(
        model_config,
        precision=training_args.precision,
        attn_implementation=training_args.attn_implementation,
        device_map=device,
        load_in_8bit=training_args.load_in_8bit,
        load_in_4bit=training_args.load_in_4bit,
    )
    model = model.eval()

    # build evaluator
    data_args = copy.deepcopy(training_args)
    data_args.dataset = evaluate_datasets[training_args.evaluate_on]
    evaluator = evaluate_class[training_args.evaluate_on](data_args)

    if training_args.library_id:
        # Add only the experts we need
        test_tasks = set(
            getattr(evaluator.datamodule, f"{training_args.split}_dataset")[
                "document_id"
            ]
        )
        expert_lib = ExpertLibrary.get_expert_library(
            training_args.library_id, selection=list(evaluator.datamodule.task_names)
        )
        model.add_experts_from_library(expert_lib)

    # We first try to load a QA expert if provided
    if training_args.ke_hf_path:
        ke_expert = load_expert(training_args.ke_hf_path)
        model.add_expert_instance(ke_expert, expert_name=training_args.ke_expert_name)

    result = evaluator.evaluate(
        model, split=args.split, shuffle=True, output_path=args.output_dir
    )
    if is_main_process():
        print(f"Result: {result}")


if __name__ == "__main__":
    args = QAEvalArguments.parse()
    eval_qa(args)
