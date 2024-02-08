import os
import sys
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.mmlu_data_module import MMLUDataConfig
from mttl.evaluators import MMLUEvaluator
from mttl.utils import remote_login, setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import (
    MoETrainer,
)
from projects.wiki_experts.src.config import ExpertConfig


def run_eval(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)
    # select dataloader
    configuration = os.environ.get("MMLU_CONFIG", None)
    logger.info("MMLU Configuration: {}".format(configuration))

    config = MMLUDataConfig(
        model=args.model,
        model_family=args.model_family,
        max_input_length=args.max_input_length,
        finetune_task_name=args.finetune_task_name,
        few_shot=args.eval_mmlu_few_shot,
        predict_batch_size=args.predict_batch_size,
    )

    mmlu = MMLUEvaluator(config)
    module = MoETrainer.load_from_checkpoint(args.checkpoint).to("cuda")
    scores = mmlu.evaluate(module, split=args.mmlu_test_split, shuffle=True)
    logger.info("MMLU scores: {}".format(scores))


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
