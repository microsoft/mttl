import json
import logging
from dataclasses import dataclass

from lightning_fabric import seed_everything
from train_qa import KEArguments

from mttl.logging import setup_logging
from mttl.models.containers.selectors.km_selector import (
    KnowledgeExtractorSelectorConfig,
)
from mttl.models.expert_model import MoEModel
from mttl.models.km_model import KMMoEModelConfig
from mttl.models.library.expert import load_expert
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import remote_login

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

    # Build model (will have 0 experts if `library_id` is None)
    model_config = KMMoEModelConfig(
        base_model=training_args.model,
        library_id=None,
        expert_selection=args.finetune_task_name,
        selector_config=KnowledgeExtractorSelectorConfig(),
    )

    # create a model without any experts
    model = MoEModel(model_config)

    from nqa_evaluator import NQAZeroShotEvaluator

    evaluator = NQAZeroShotEvaluator(training_args, generation_kwargs={})

    if training_args.library_id:
        # Add only the experts we need
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

    if args.nqa_dataset is None:
        logger.info(f"Setting callback dataset to {args.dataset}")
        args.nqa_dataset = args.dataset

    # Allow to set trainable tasks from a json split file (e.g. nqa_mini_split.json)
    if isinstance(args.finetune_task_name, str) and args.finetune_task_name.endswith(
        ".json"
    ):
        with open(args.finetune_task_name, "r") as f:
            split_dict = json.load(f)

        tasks = split_dict["test"]
        logger.info(f"Setting finetune_task_name to {tasks}")
        args.finetune_task_name = tasks

    eval_qa(args)
