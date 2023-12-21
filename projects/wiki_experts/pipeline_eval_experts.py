import os
import sys
from huggingface_hub import login
from pytorch_lightning import seed_everything

from mttl.datamodule.openbookqa_data_module import OpenbookQADataConfig
from mttl.evaluators.openbookqa_evaluator import OpenbookQAEvaluator

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.hswag_data_module import HellaswagDataConfig
from mttl.datamodule.humaneval_module import HumanEvalConfig
from mttl.datamodule.mbpp_datamodule import MBPPDataConfig
from mttl.datamodule.arc_data_module import ArcDataConfig
from mttl.datamodule.piqa_data_module import PiqaDataConfig
from mttl.datamodule.winogrande_data_module import WinograndeDataConfig
from mttl.evaluators.winogrande_evaluator import WinograndeEvaluator
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.datamodule.bbh_data_module import BBHConfig
from mttl.evaluators.arc_evaluator import ArcEvaluator
from mttl.evaluators.piqa_evaluator import PiqaEvaluator
from mttl.evaluators.hellaswag_evaluator import HellaswagEvaluator
from mttl.evaluators.humaneval_evaluator import HumanEvalEvaluator
from mttl.evaluators.mbpp_evaluator import MBPPEvaluator
from mttl.evaluators.bbh_evaluator import BBHEvaluator
from mttl.utils import setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    MultiExpertModelRanker,
)
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.mmlu_eval_experts import parse_experts_to_load


def setup_evaluators(args, active_tasks=["piqa"]):
    evaluators = {}
    common_kwargs = {
        "model": args.model,
        "model_family": args.model_family,
        "max_input_length": args.max_input_length,
        "max_output_length": args.max_output_length,
        "predict_batch_size": args.predict_batch_size,
        "truncation_side": args.truncation_side,
    }
    generation_kwargs = {
        "temperature": 0.05,
        "top_p": 0.95,
        "do_sample": True,
    }

    for task in set(active_tasks):
        if task == "humaneval":
            common_kwargs["max_output_length"] = 300
            config = HumanEvalConfig(
                **common_kwargs,
            )
            evaluators["humaneval"] = HumanEvalEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task == "mbpp":
            common_kwargs["max_output_length"] = 300
            evaluators["mbpp"] = MBPPEvaluator(
                MBPPDataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        elif task == "bbh":
            config = BBHConfig(
                **common_kwargs,
            )
            evaluators.append["bbh"] = BBHEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task == "arc-easy":
            config = ArcDataConfig(
                **common_kwargs,
                arc_type="ARC-Easy",
            )
            evaluators["arc-easy"] = ArcEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task == "arc-challenge":
            config = ArcDataConfig(
                **common_kwargs,
                arc_type="ARC-Challenge",
            )
            evaluators["arc-challenge"] = ArcEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task == "piqa":
            config = PiqaDataConfig(
                **common_kwargs,
            )
            evaluators["piqa"] = PiqaEvaluator(
                config, generation_kwargs=generation_kwargs
            )
        elif task == "hswag":
            evaluators["hswag"] = HellaswagEvaluator(
                HellaswagDataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        elif task == "winogrande":
            evaluators["winogrande"] = WinograndeEvaluator(
                WinograndeDataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        elif task == "openbookqa":
            evaluators["openbookqa"] = OpenbookQAEvaluator(
                OpenbookQADataConfig(**common_kwargs),
                generation_kwargs=generation_kwargs,
            )
        else:
            raise ValueError("No active tasks")
    return evaluators


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
        library = HFExpertLibrary(args.hf_lib_id)
        logger.info("Loaded library: {}".format(library))
    else:
        library = None

    if args.load_module is not None:
        kwargs = parse_experts_to_load(args.load_module)
        for expert_kwargs in kwargs:
            module.load_expert(**expert_kwargs, expert_library=library)
    elif args.module_graph is not None:
        module.load_from_graph_string(args.module_graph, expert_library=library)
    module.to("cuda")

    evaluators = setup_evaluators(
        args, active_tasks=args.pipeline_eval_tasks.split(",")
    )

    for name, evaluator in evaluators.items():
        scores = evaluator.evaluate(module, shuffle=True)
        logger.info("Task: {}".format(name))
        logger.info("Scores: {}".format(scores))

        with open(args.output_dir + f"/{name}_scores.json", "w") as f:
            import json

            json.dump(scores, f)


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
