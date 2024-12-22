# here, we train experts and we upload them to a local library (repository) of experts.

import os
from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.expert_model import (
    ExpertModel,
    MultiExpertModel,
    MultiExpertModelConfig,
    ExpertModelConfig,
)
from mttl.models.train_utils import train_model

from mttl.evaluators.gsm_evaluator import GsmEvaluator
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.arguments import EvaluationConfig, ExpertConfig
from mttl.models.lightning.expert_module import MultiExpertModule
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

args = EvaluationConfig.parse()

datamodule = get_datamodule(args, for_generation=True)
evaluator = GsmEvaluator(datamodule)

module = MultiExpertModule(model=args.model).to(device)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)["state_dict"]
    module.load_state_dict(checkpoint)
## evaluate
result = evaluator.evaluate(module.model, split="test")
print(result)
