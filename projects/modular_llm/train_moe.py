from mttl.config import MoEExpertConfig
from mttl.models.expert_model import MoEModel
from projects.modular_llm.train_experts import train_experts

if __name__ == "__main__":
    train_experts(MoEExpertConfig.parse(), MoEModel)
