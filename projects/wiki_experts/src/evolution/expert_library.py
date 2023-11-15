import glob
import torch
import numpy as np
from collections import defaultdict


class ExpertLibrary:
    """
    Searches local experts
    """

    def __init__(self, modules_dir=None, selection="-val", operator=np.argmin):
        self.home_dir = modules_dir
        # searches home and loads all the existing experts with selection criteria

        all_checkpoints = glob.glob(f"{self.home_dir}/**/*{selection}/*.ckpt")
        all_checkpoints += glob.glob(f"{self.home_dir}/**/**/*{selection}/*.ckpt")

        self.operator = operator
        # expert per model and task
        self.experts = defaultdict(lambda: defaultdict(list))
        for path in all_checkpoints:
            ckpt = torch.load(path, map_location="cpu")
            model = ckpt["hyper_parameters"]["model"]
            task = ckpt["hyper_parameters"]["finetune_task_name"]
            self.experts[model][task].append(path)

        print(
            f"Found {len(all_checkpoints)} experts in {self.home_dir} for models {list(self.experts.keys())}"
        )
        # adding base module also
        for model in self.experts.keys():
            self.experts[model]["base"] = [None]

    def get_all_tasks(self):
        return [t for m in self.experts.values() for t in m.keys()]

    def get_experts_for_model(self, model):
        assert model in self.experts, f"Model {model} not found in {self.home_dir}"
        return list(self.experts[model].keys())

    def get_experts(self, model, task):
        assert model in self.experts, f"Model {model} not found in {self.home_dir}"
        return self.experts[model][task]

    def get_expert_path(self, model, task):
        experts = self.get_experts(model, task)
        if task == "base":
            return None
        metrics = [
            float(e.split("/")[-1].replace(".ckpt", "").replace("loss=", ""))
            for e in experts
        ]
        args_best = self.operator(metrics)
        return experts[args_best]
