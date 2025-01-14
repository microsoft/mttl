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
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.arguments import EvaluationConfig, ExpertConfig
from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
import torch
from mttl.logging import setup_logging

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_logging()

args = EvaluationConfig.parse()

datamodule = get_datamodule(args, for_generation=True)
evaluator = GsmEvaluator(datamodule)

if args.library_id is None:
    module = ExpertModule(**vars(args)).to(device)
else:
    module = MultiExpertModule(**vars(args)).to("cuda")
    module.add_experts_from_library(args.library_id)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
    module.load_state_dict(checkpoint)
## evaluate
result = evaluator.evaluate(module.model, split="test")
