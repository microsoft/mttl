import os
import sys
from pytorch_lightning import seed_everything
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from mttl.utils import setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import (
    MultiExpertModelRanker,
    MultiExpertModelClipRanker,
)
from projects.wiki_experts.src.config import ExpertConfig
from mttl.evaluators.rouge_evaluator import RougeEvaluator


def parse_experts_to_load(experts_to_load):
    kwargs = []

    def find_experts(path):
        import glob

        for path in glob.glob(expert_path + "/**/csv_metrics/", recursive=True):
            yield "/".join(path.split("/")[:-2])

    if type(experts_to_load) != list:
        experts_to_load = [experts_to_load]

    for expert in experts_to_load:
        options = expert.split(":")
        expert_path = options[0]
        expert_path, _, expert_name = expert_path.partition("=")
        all_paths = list(find_experts(expert_path)) or [expert_path]

        if not expert_name:
            expert_name = None

        if len(options) >= 2:
            action = options[1]
        else:
            action = "route"

        if len(options) >= 3:
            load_only_layers = options[2]
        else:
            load_only_layers = None

        is_default = "*" in action
        action = action.replace("*", "")

        if len(all_paths) > 1:
            if is_default:
                raise ValueError(
                    "Cannot define more than one default expert! Are you using * in expert path?"
                )
            if expert_name:
                raise ValueError(
                    "Cannot declare a name when using a wildcard in the expert path!"
                )

        kwargs.append(
            {
                "expert_path": expert_path,
                "action": action,
                "is_default": is_default,
                "expert_name": expert_name,
                "load_only_layers": load_only_layers,
            }
        )
    return kwargs


def run_eval(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    # select dataloader
    configuration = os.environ.get("FLAN_CONFIG", None)
    logger.info("flan tasks names: {}".format(configuration))
    # add FlanEvaluator
    data_module = FlanModule(
        FlanConfig(
            dataset="sordonia/flan-10k-flat",
            model="EleutherAI/gpt-neo-125m",
            finetune_task_name=configuration,
            predict_batch_size=8,
        ),
        for_generation=True,
    )

    evaluator = RougeEvaluator(data_module)
    # load module
    if args.retrieval_model == "clip":
        print("Using clip model")
        module = MultiExpertModelClipRanker(
            **vars(args),
            tokenizer=data_module.tokenizer,
        )
    else:
        print("Using classifier model")
        module = MultiExpertModelRanker(
            **vars(args),
            tokenizer=data_module.tokenizer,
        )
    if args.expert_library_path:
        library = HFExpertLibrary(args.expert_library_path)
        module.load_from_library(library)
    elif args.load_module is not None:
        kwargs = parse_experts_to_load(args.load_module)
        for expert_kwargs in kwargs:
            module.load_expert(**expert_kwargs)
    elif args.module_graph is not None:
        module.load_from_graph_string(args.module_graph)

    module.to("cuda")
    # evaluate all the category
    rouge = evaluator.evaluate(module, split="test", verbose=False)

    logger.info("Flan rouge: {}".format(rouge))
    del module


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
