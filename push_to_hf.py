from transformers import AutoTokenizer, AutoModelForCausalLM
from mttl.arguments import EvaluationConfig
from mttl.models.lightning.expert_module import ExpertModule, MultiExpertModule
from mttl.models.containers.selectors.base import UniformSelectorConfig
from mttl.models.library.expert_library import ExpertLibrary
import torch
from mttl.logging import setup_logging
setup_logging()
args = EvaluationConfig.parse()
if args.library_id is None:
    module = ExpertModule(**vars(args))
else:

    library = ExpertLibrary.get_expert_library(args.library_id)
    module = MultiExpertModule(**vars(args))

    if args.merge_or_route == "uniform":
        module.add_experts_from_library(args.library_id)
        module.model.set_selector("lora", UniformSelectorConfig(lora_merge_after=args.lora_merge_after))
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
    module.load_state_dict(checkpoint)


if args.save_merged_model:
    module.model.merge_and_save_base_model(args.output_dir)
    # clear the gpu memory

    torch.cuda.empty_cache()
    del module
    del checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    model = AutoModelForCausalLM.from_pretrained(args.output_dir)
    tokenizer.push_to_hub(args.save_merged_model)
    model.push_to_hub(args.save_merged_model)


