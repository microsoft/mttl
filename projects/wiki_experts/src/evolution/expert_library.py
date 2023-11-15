import glob
import torch
import numpy as np
from collections import UserDict
from collections import defaultdict


class ExpertLibrary(UserDict):
    def __init__(self, modules_dir, model_name, selection="", operator="mean"):
        """
        Searches local experts
        """
        super().__init__()
        self.home_dir = modules_dir
        # searches home and loads all the existing experts with selection criteria
        all_checkpoints = glob.glob(f"{self.home_dir}/**/*{selection}/*.ckpt")
        all_checkpoints += glob.glob(f"{self.home_dir}/**/**/*{selection}/*.ckpt")
        self.model_name = model_name

        self.operator = operator
        # expert per model and task
        for path in all_checkpoints:
            ckpt = torch.load(path, map_location="cpu")
            model = ckpt["hyper_parameters"]["model"]
            if model != self.model_name:
                continue
            task = ckpt["hyper_parameters"]["finetune_task_name"]
            if task not in self.__dict__:
                self.data[task] = []
            self.data[task].append(path)

        print(
            f"Found {len(all_checkpoints)} experts in {self.home_dir} for models {list(self.__dict__.keys())}"
        )
        # adding base module also
        self.data["base"] = [None]
