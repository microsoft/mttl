import copy
import json
import logging
import os
import time
from dataclasses import dataclass

from lightning_fabric import seed_everything

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule
from projects.kms.utils.longhealth_datamodule import LonghealthDatamodule
from projects.kms.utils.nqa_datamodule import NQADatamodule
from projects.kms.utils.quality_datamodule import (
    GenQualityDataModule,
    QualityDatamodule,
)

# isort: split
from mttl.arguments import ExpertConfig, MultiExpertConfig
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.logging import logger, setup_logging
from mttl.models.containers.selectors import TaskNameSelectorConfig
from mttl.models.expert_model import MoEModel, MultiExpertModel, MultiExpertModelConfig
from mttl.models.library.expert import load_expert
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import remote_login
from projects.kms.train_km_simple import (
    evaluate_class,
    evaluate_datasets,
    evaluate_metrics,
)
from projects.kms.train_qa import cpu_offload
from projects.kms.utils.km_model import KEMoEModelConfig
from projects.kms.utils.km_selector import KnowledgeExtractorSelectorConfig


@dataclass
class QAEvalArguments(MultiExpertConfig):
    # split
    split: str = "test"
    # Which datamodule to use
    evaluate_on: str = None
    # whether to print evluator input / output
    verbose: bool = False
    # maybe load a trained KE
    ke_uri: str = None


def eval_qa(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token, raise_error=False)

    # We want to support 3 use-cases
    # 1) Only the KM is present --> Resort to TaskNameSelector (default)
    # 2) Only the KE is present --> Resort to DefaultExpert
    # 3) Both are present --> Resort to KE
    if training_args.library_id and training_args.ke_uri:
        selector_class = KnowledgeExtractorSelectorConfig
    else:
        selector_class = TaskNameSelectorConfig

    device = get_device()
    model_config = MultiExpertModelConfig(
        base_model=training_args.model,
        selector_config=selector_class.from_training_config(training_args),
    )
    model = MultiExpertModel(
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
    metric = evaluate_metrics[training_args.evaluate_on]

    if training_args.library_id:
        # Add only the experts we need
        expert_selection = getattr(
            evaluator.datamodule, f"{training_args.split}_task_names"
        )
        expert_lib = ExpertLibrary.get_expert_library(
            training_args.library_id, selection=expert_selection
        )
        model.add_experts_from_library(expert_lib)
        print("Loaded: ", len(model.experts_names))

    if training_args.ke_uri:
        ke_expert = load_expert(training_args.ke_uri)
        model.add_expert_instance(ke_expert, expert_name="KE")
        if training_args.library_id is None:
            # set the default expert to the KE
            model.set_default_expert("KE")

    result = evaluator.evaluate(
        model,
        split=args.split,
        shuffle=True,
        output_path=args.output_dir,
        verbose=args.verbose,
    )
    if is_main_process():
        print(f"{metric}: {result}")


if __name__ == "__main__":
    args = QAEvalArguments.parse()

    eval_qa(args)
