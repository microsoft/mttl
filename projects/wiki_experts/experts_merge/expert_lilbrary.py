import glob
import torch
import numpy as np
from collections import defaultdict
from mttl.utils import setup_logging, logger


def get_file(base_dir, subject=None, selection="_test_oracle", operator=np.argmin):
    if subject:
        subject_dirs = [
            f"{base_dir}/**/{subject}/*{selection}",
            f"{base_dir}/**/**/{subject}/*{selection}",
        ]
        subject_dirs += [
            f"{base_dir}/**/{subject}/*{selection}/**/",
            f"{base_dir}/**/**/{subject}*{selection}/**/",
        ]
        files = []
        for subject_dir in subject_dirs:
            files += glob.glob(f"{subject_dir}/*.ckpt")
        if len(files) == 0:
            logger.warning(f"no ckpt files found for {subject}")
            return
        best_idx = operator(
            [
                float(f.split("/")[-1].replace(".ckpt", "").replace("loss=", ""))
                for f in files
            ]
        )
        file = files[best_idx]
        # for f in files:
        #     if f!=file:
        #         os.remove(f)
        return file
    else:
        dirs = [f"{base_dir}/**/*{selection}", f"{base_dir}/**/**/*{selection}"]
        files = glob.glob(f"{dirs[0]}/*.ckpt") + glob.glob(f"{dirs[1]}/*.ckpt")
        if len(files) == 0:
            logger.warning(f"no ckpt files found for {subject}")
            return
        best_idx = operator(
            [
                float(f.split("/")[-1].replace(".ckpt", "").replace("loss=", ""))
                for f in files
            ]
        )
        file = files[best_idx]
        return file


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
