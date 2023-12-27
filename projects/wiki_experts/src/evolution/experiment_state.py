import os
import sys
import copy
import torch
import wandb
from dataclasses import dataclass
from config import EvolExpertConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from projects.wiki_experts.src.evolution.utils import TableLogger
from mttl.models.modifiers.expert_containers.expert_library import LocalExpertLibrary


class ExperimentState:
    @dataclass
    class State:
        config: EvolExpertConfig
        active_iteration: int
        expert_lib: LocalExpertLibrary
        results_table: TableLogger

    def __init__(self, **kwargs):
        self.state = self.State(**kwargs)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.state, k, v)

    @property
    def path(self):
        if wandb.run is not None:
            exp_name = wandb.run.name
        else:
            exp_name = os.getenv("AMLT_JOB_NAME", "_some_experiment")
        exp_name = exp_name.replace("/", "_")
        path = self.state.config.output_dir
        path = os.path.join(path, f"exp_state_{exp_name}")

        os.makedirs(path, exist_ok=True)
        return path

    def save(self, path=None):
        path = path or self.path
        if not path.endswith(".pt"):
            path = os.path.join(path, "state.pt")

        state = copy.deepcopy(self.state)
        torch.save(state, path)

    def load_from_path(self, path=None):
        path = path or self.path
        if not path.endswith(".pt"):
            path = os.path.join(path, "state.pt")
        state = torch.load(path)
        self.state = state
