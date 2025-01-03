import copy
import json
import logging
from dataclasses import dataclass

from lightning_fabric import seed_everything

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule
from projects.kms.utils.nqa_datamodule import NQADatamodule

# isort: split

from mttl.logging import setup_logging
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
from projects.kms.train_qa import KEArguments, train_datasets
from projects.kms.utils.nqa_evaluator import NQAZeroShotEvaluator
from projects.kms.utils.quality_evaluator import QualityEvaluator
from projects.kms.utils.wiki_mmlu_evaluator import WikiMMLUEvaluator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class QAEvalArguments(KEArguments):
    ke_expert_name: str = "KE"
    split: str = "test"


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
    model = MoEModel(model_config)

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
            training_args.library_id, selection=list(test_tasks)
        )
        model.add_experts_from_library(expert_lib)

    # We first try to load a QA expert if provided
    if training_args.ke_hf_path:
        ke_expert = load_expert(training_args.ke_hf_path)
        model.add_expert_instance(ke_expert, expert_name=training_args.ke_expert_name)

    model = model.cuda()

    # Call the NQA callback
    rougeL = evaluator.evaluate(model, split=args.split)

    print(f"ROUGE-L: {rougeL}")


if __name__ == "__main__":
    args = QAEvalArguments.parse(raise_error=False)

    """
    if args.nqa_dataset is None:
        logger.info(f"Setting callback dataset to {args.dataset}")
        args.nqa_dataset = args.dataset

    # Callback actually reads from `args.dataset`
    args.dataset = args.nqa_dataset
    """
    if args.use_rag:
        if not args.evaluate_on.endswith("-rag"):
            logger.warning(f"Overwriting `evaluate_on` to {args.evaluate_on}-rag")
            args.evaluate_on += "-rag"
        args.include_context = True
    else:
        args.include_context = False

    args.dataset = train_datasets[args.evaluate_on]
    dataset = args.evaluate_on.split("-")[0]
    dataset_type = {"nqa": "narrativeqa", "quality": "quality"}[dataset]
    if args.dataset_type != dataset_type:
        logger.warning(f"Overwriting `dataset_type` to {dataset_type}")
        args.dataset_type = dataset_type

    if dataset == "quality" and args.split != "dev":
        logger.warning(
            f"Quality has not labelled test split. Overwriting `split` to dev"
        )
        args.split = "dev"
        args.subsample_dev = args.subsample_test

    eval_on = args.evaluate_on.split("-")[0]
    if args.finetune_task_name is None:
        args.finetune_task_name = {
            "quality": "splits/quality/quality_full.json",
            "nqa": "splits/nqa/nqa_full.json",
        }[eval_on]
        logger.warning(f"Overwriting `finetune_task_name` to {args.finetune_task_name}")

    # Allow to set trainable tasks from a json split file (e.g. nqa_mini_split.json)
    if isinstance(args.finetune_task_name, str) and args.finetune_task_name.endswith(
        ".json"
    ):

        if args.subsample_file and args.subsample_file != args.finetune_task_name:
            raise ValueError(
                "Cannot have different subsample_file and finetune_task_name"
            )

        with open(args.finetune_task_name, "r") as f:
            split_dict = json.load(f)

        tasks = split_dict[args.split]
        logger.info(f"Setting finetune_task_name to {tasks}")
        args.finetune_task_name = tasks

    eval_qa(args)
