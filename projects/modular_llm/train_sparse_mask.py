from mttl.arguments import ExpertConfig
from mttl.models.lightning.expert_module import SPLITExpertModule
from projects.modular_llm.train_experts import train_experts

if __name__ == "__main__":
    train_experts(ExpertConfig.parse(), SPLITExpertModule)
