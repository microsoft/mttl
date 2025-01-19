# here, we train experts and we upload them to a local library (repository) of experts.

from mttl.datamodule.base import get_datamodule
from mttl.models.library.expert_library import ExpertLibrary

from mttl.evaluators.gsm_evaluator import GsmEvaluator
from mttl.arguments import EvaluationConfig
from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
import torch
from mttl.logging import setup_logging

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_logging()

args = EvaluationConfig.parse()


from mttl.datamodule.gsm_data_module import GsmDataConfig, GsmDataModule

# datamodule = get_datamodule(args, for_generation=True)

config = GsmDataConfig(
    model=args.model,
    model_family=args.model_family,
    max_input_length=args.max_input_length,
    max_output_length=args.max_output_length,
    gsm_template=args.gsm_template,
    data_dir=args.output_dir,
    few_shot=args.few_shot,
)
dm = GsmDataModule(config, for_generation=True)

evaluator = GsmEvaluator(dm)

if args.library_id is None:
    module = ExpertModule(**vars(args)).to(device)
else:

    library = ExpertLibrary.get_expert_library(args.library_id)
    module = MultiExpertModule(**vars(args)).to(device)
    if args.expert_selection is not None:
        expert = library.get_expert(args.expert_selection)
        module.add_expert_instance(expert, is_default=True)
    else:
        module = MultiExpertModule(**vars(args)).to(device)
        module.add_experts_from_library(args.library_id)

if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
    module.load_state_dict(checkpoint)
## evaluate
result = evaluator.evaluate(module.model, split="test")
