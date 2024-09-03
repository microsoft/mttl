from mttl.arguments import MoEExpertConfig
from mttl.models.lightning.expert_module import MoEModule
from projects.modular_llm.train_experts import train_experts

if __name__ == "__main__":
    train_experts(MoEExpertConfig.parse(), MoEModule)
