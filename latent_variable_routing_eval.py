# here, we train experts and we upload them to a local library (repository) of experts.

from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.evaluators.gsm_evaluator import GsmEvaluator
from mttl.arguments import EvaluationConfig
from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
import torch
from tqdm import tqdm
from mttl.logging import setup_logging
import numpy as np
import matplotlib.pyplot as plt
import json
from mttl.models.library.library_transforms import WudiMerge, WudiMergeConfig
from mttl.datamodule.base import get_datamodule
from mttl.datamodule.abstention_data_module import AbstentionDataModule
from mttl.datamodule.task_adapter_data_module import TaskAdapterModule, TaskAdapterConfig
from mttl.evaluators.abstain_evaluator import AbstainQAEvaluator
from mttl.evaluators.asr_evaluator import ASREvaluator

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_logging()

args = EvaluationConfig.parse()


if args.library_id is None:
    module = ExpertModule(**vars(args)).to(device)
else:

    library = ExpertLibrary.get_expert_library(args.library_id)
    module = MultiExpertModule(**vars(args)).to(device)
    if args.expert_selection is not None:
        expert = library.get_expert(args.expert_selection)
        module.add_expert_instance(expert, is_default=True)
    else:
        module.add_experts_from_library(args.library_id)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
    module.load_state_dict(checkpoint)


# config = TaskAdapterConfig(model=args.model, finetune_task_name=args.finetune_task_name, max_output_length=args.max_output_length,)
# dm_for_gen = TaskAdapterModule(config, for_generation=True)

# abstainqa_evaluator = AbstainQAEvaluator(
#     datamodule=dm_for_gen
# )
# abstain_scores = abstainqa_evaluator.evaluate(module.model, split="test", verbose=False)

dm = get_datamodule(args, for_generation=True)
asr_evaluator = ASREvaluator(
    datamodule=dm
)
asr_scores = asr_evaluator.evaluate(module.model, split="test", verbose=False)
print(f"ASR: {asr_scores}")





