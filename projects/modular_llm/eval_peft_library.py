from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.arguments import ExpertConfig
from mttl.datamodule.abstention_data_module import (
    AbstentionDataConfig,
    AbstentionDataModule,
)

from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
import torch
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.evaluators.rouge_evaluator import RougeEvaluator, GenerativeEvaluator
from mttl.logging import setup_logging
from mttl.arguments import EvaluationConfig

setup_logging()

train_config = ExpertConfig.from_dict(
    {
        "lora_rank": 4,
        "lora_alpha": 1.0,
        "lora_dropout": 0.0,
        "weight_decay": 0.0,
        "output_dir": "/tmp/",
        "model_modifier": "lora",
        "modify_modules": ".*",
        "modify_layers": "q_proj|v_proj|k_proj",
        "trainable_param_names": ".*lora_[ab].*",
        "num_train_epochs": 5,
        "learning_rate": 1e-2,
        "micro_batch_size": 16,
        "train_batch_size": 16,
        "predict_batch_size": 8,
        "precision": "bf16",
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_family": "gpt",
        "optimizer": "adamw",
        "dataset": "sordonia/flan-100-flat",
        "warmup_proportion": 0.0,
        "max_input_length": 1024,
        "max_output_length": 128,
        "truncation_side": "left",
    }
)

args = EvaluationConfig.parse()
device_map = "cuda" if torch.cuda.is_available() else "cpu"
model = None
if args.merge_or_route == "base":
    # base model
    model = MultiExpertModel(
        MultiExpertModelConfig(base_model="mistralai/Mistral-7B-Instruct-v0.3"),
        device_map=device_map,
        precision="32",
    )
elif args.merge_or_route == "uniform":
    model = MultiExpertModel.from_pretrained_library(
        args.library_id,
        selector_config=UniformSelectorConfig(),
        device_map=device_map,
    )
else:
    module = MultiExpertModule(**vars(args)).to("cuda")
    module.add_experts_from_library(args.library_id)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
        module.load_state_dict(checkpoint)
    model = module.model
    # module.push_to_hub("zhan1993:/")

## load datasets
config = AbstentionDataConfig(
    model=args.model,
    dataset=args.dataset,  # zhan1993/coconot_original_eval
    predict_output_dir=args.output_dir,
)
datamodule = AbstentionDataModule(config, for_generation=True)


evaluator = RougeEvaluator(datamodule=datamodule)
evaluator.generate(model, split="test", verbose=False)

# train_dataloader = datamodule.test_dataloader()
# for batch in train_dataloader:
#     batch = transfer_batch_to_device(batch, device_map)
#     print(batch)
#     breakpoint()


# set tasks

# from mttl.evaluators.base import EvaluatorRunner, setup_evaluators

# # we will run on bbh-fast
# evaluation_task = "bbh-fast"
# with torch.no_grad():
#     runner: EvaluatorRunner = setup_evaluators(
#         model_type=train_config.model,
#         model_family=train_config.model_family,
#         max_input_length=train_config.max_input_length,
#         max_output_length=train_config.max_output_length,
#         predict_batch_size=train_config.predict_batch_size,
#         truncation_side=train_config.truncation_side,
#         tasks=evaluation_task,
#     )

# base_perf = runner.run(model)
# print(base_perf)
