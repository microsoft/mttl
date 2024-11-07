from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig

model = MultiExpertModel.from_pretrained_library(
    "sordonia/library-kms-7449cf51",
    device_map="cpu",
    precision="bf16",
)
breakpoint()
