import os
import sys
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from mttl.models.modifiers.routing import RoutingInfo
from mttl.models.modifiers import modify_transformer
from mttl.models.modifiers.prompt_tuning import PromptTuning, PromptTuningConfig
from projects.wiki_experts.src.evolution.nevergrad_opt import NGRoutingOptimizer


def test_train_router():
    return


def test_NGRoutingOptimizer():
    return
