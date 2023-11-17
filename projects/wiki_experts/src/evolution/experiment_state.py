import os
import sys
import copy
import torch
import wandb
from dataclasses import dataclass
from config import ExpertsMergeConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from projects.wiki_experts.src.evolution.utils import TableLogger
from projects.wiki_experts.src.expert_library import LocalExpertLibrary


class ExperimentState:
    @dataclass
    class State:
        config: ExpertsMergeConfig
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
            run_name = wandb.run.name
        else:
            run_name = os.getenv("AMLT_JOB_NAME", "_some_experiment")
        run_name = run_name.replace("/", "_")
        path = self.state.config.output_dir
        path = os.path.join(path, f"exp_state_{run_name}")

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
