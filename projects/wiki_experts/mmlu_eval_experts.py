import os
import sys
from pytorch_lightning import seed_everything
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary
from mttl.models.modifiers.expert_containers.module_graph import Expert, ExpertInfo
from mttl.models.modifiers.hard_prompts import HardPrompt, HardPromptConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.mmlu_data_module import MMLUDataConfig
from mttl.evaluators import MMLUEvaluator
from mttl.utils import remote_login, setup_logging, logger

# register models
from projects.wiki_experts.src.expert_model import (
    MultiExpertModel,
    MultiExpertModelRanker,
)
from projects.wiki_experts.src.config import ExpertConfig


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
            }
        )
    return kwargs


def run_eval(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    remote_login(args.remote_token)
    # select dataloader
    configuration = os.environ.get("MMLU_CONFIG", None)
    logger.info("MMLU Configuration: {}".format(configuration))

    if configuration == "random_5":
        args.finetune_task_name = "college_biology,high_school_government_and_politics,prehistory,security_studies"
        subsample = None
    elif configuration == "worst_5":
        args.finetune_task_name = "formal_logic,machine_learning,global_facts,abstract_algebra,high_school_physics"
        subsample = None
    elif configuration == "first":
        args.finetune_task_name = "abstract_algebra"
        subsample = None
    elif configuration == "sub_10":
        args.finetune_task_name = "formal_logic,machine_learning,global_facts,abstract_algebra,high_school_physics,college_biology,high_school_government_and_politics,prehistory,security_studies,sociology"
        subsample = None
    else:
        subsample = None

    config = MMLUDataConfig(
        model=args.model,
        model_family=args.model_family,
        max_input_length=args.max_input_length,
        finetune_task_name=args.finetune_task_name,
        few_shot=args.eval_mmlu_few_shot,
        predict_batch_size=args.predict_batch_size,
    )
    mmlu = MMLUEvaluator(config)

    # load module
    if args.ranker_model is not None:
        module = MultiExpertModelRanker(
            **vars(args), tokenizer=mmlu.datamodule.tokenizer
        )
    else:
        module = MultiExpertModel(**vars(args), tokenizer=mmlu.datamodule.tokenizer)

    if args.hf_lib_id:
        library = HFExpertLibrary(args.hf_lib_id)
    else:
        library = None

    if args.load_module is not None:
        kwargs = parse_experts_to_load(args.load_module)
        for expert_kwargs in kwargs:
            module.load_expert(**expert_kwargs, expert_library=library)
    elif args.module_graph is not None:
        module.load_from_graph_string(args.module_graph, expert_library=library)

    if args.mmlu_use_hard_prompt:
        config = HardPromptConfig(
            max_input_length=args.max_input_length,
            model_family=args.model_family,
            tokenizer=module.tokenizer,
        )
        expert = Expert(
            expert_info=ExpertInfo("hard_prompt", expert_config=config),
            expert_weights=args.mmlu_use_hard_prompt,
        )
        module.add_expert_instance(expert, action="route", is_default=True)

    module.to("cuda")
    scores = mmlu.evaluate(
        module, split=args.mmlu_test_split, subsample=subsample, shuffle=True
    )
    logger.info("MMLU Accuracy: {}".format(scores))


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
