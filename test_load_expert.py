from mttl.models.library.expert_library import ExpertLibrary
from mttl.logging import setup_logging
from mttl.models.library.expert import Expert, load_expert
from mttl.arguments import EvaluationConfig
import torch
from mttl.models.lightning.expert_module import ExpertModule
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--library_id",
    type=str,
    default="trained_gpt125m_experts_colab",
    help="ID of the expert library"
)
parser.add_argument(
    "--checkpoint_1",
    type=str,
    required=True,
    help="Path to first expert checkpoint"
)
parser.add_argument(
    "--checkpoint_2", 
    required=True,
    type=str,
    help="Path to second expert checkpoint"
)

args = parser.parse_args()



library = ExpertLibrary.get_expert_library(f"local://{args.library_id}", create=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
setup_logging()

expert_1 = load_expert(
            args.checkpoint_1,
            expert_name="expert1",
)

expert_2 = load_expert(
            args.checkpoint_1,
            expert_name="expert2",
)

library.add_expert(expert_1, force=True)
library.add_expert(expert_2, force=True)

