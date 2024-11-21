from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.models.lightning.expert_module import ExpertModule
from mttl.models.lightning.expert_module import (
    MultiExpertModule,
    MultiExpertModelConfig,
)
from mttl.models.library.expert_library import ExpertLibrary
from mttl.arguments import ExpertConfig
from transformers import AutoTokenizer
import torch

args = ExpertConfig.parse()
args.model = "EleutherAI/gpt-neo-125m"
tokenizer = AutoTokenizer.from_pretrained(args.model)

## test expert model
model = ExpertModule(**vars(args))
model1_dict = model.state_dict()

checkpoint = torch.load(args.checkpoint, weights_only=False)["state_dict"]
model.load_state_dict(checkpoint)

model2_dict = model.state_dict()

breakpoint()

are_equal = all(
    torch.equal(model1_dict[key], model2_dict[key])
    for key in model1_dict
    if key in model2_dict
)
print("两个模型参数是否相同:", are_equal)
breakpoint()

text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(outputs)
