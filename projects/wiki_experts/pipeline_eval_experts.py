import glob
import os
import sys

import prettytable
from huggingface_hub import login
from pytorch_lightning import seed_everything

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.evaluators.base import EvaluatorRunner, setup_evaluators
from mttl.models.modifiers.expert_containers.expert_library import (
    HFExpertLibrary,
    LocalExpertLibrary,
)
from mttl.utils import setup_logging, logger
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    MultiExpertModelRanker,
)
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.mmlu_eval_experts import parse_experts_to_load


def run_eval(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # load module
    if args.ranker_model is not None:
        module = MultiExpertModelRanker(**vars(args))
    else:
        module = MultiExpertModel(**vars(args))

    if args.hf_lib_id:
        if os.path.exists(args.hf_lib_id):
            # it's a local library
            library = LocalExpertLibrary("/tmp/experts", create=True)

            for file in glob.glob(os.path.join(args.hf_lib_id, "*")):
                library.add_expert_from_ckpt(file, force=True)
        else:
            library = HFExpertLibrary(args.hf_lib_id)

        logger.info("Loaded library: {}".format(args.hf_lib_id))
    else:
        library = None

    if args.load_module is not None:
        kwargs = parse_experts_to_load(args.load_module)
        for expert_kwargs in kwargs:
            module.load_expert(**expert_kwargs, expert_library=library)
    if args.hf_lib_id is not None:
        module.add_experts_from_library(library)
    module.to("cuda")

    runner: EvaluatorRunner = setup_evaluators(
        model_type=args.model,
        model_family=args.model_family,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        predict_batch_size=args.predict_batch_size,
        truncation_side=args.truncation_side,
        tasks=args.pipeline_eval_tasks.split(",") if args.pipeline_eval_tasks else None,
        output_path=args.output_dir,
    )
    runner.run(module)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
