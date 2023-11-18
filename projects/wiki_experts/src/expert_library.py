import glob
import io
import torch
import os
import numpy as np
from collections import UserDict

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub import (
    login,
    CommitOperationAdd,
    create_commit,
    preupload_lfs_files,
    create_repo,
)

from mttl.utils import logger, retry_with_exponential_backoff
from projects.wiki_experts.src.graph.module_graph import Expert, load_expert


class HFExpertLibrary(UserDict):
    def __init__(self, repo_id, model_name, selection=""):
        super().__init__()

        self.repo_id = repo_id
        self.model_name = model_name

        if "HF_TOKEN" in os.environ:
            login(token=os.environ["HF_TOKEN"])

        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
        except:
            pass

        if huggingface_hub.file_exists(repo_id, "library.ckpt"):
            library_ckpt = huggingface_hub.hf_hub_download(
                repo_id, filename="library.ckpt"
            )
            library_dump = torch.load(library_ckpt, map_location="cpu")

            for expert_name, expert_dump in zip(
                library_dump["expert_name"], library_dump["expert_dump"]
            ):
                self.add_expert(expert_name, expert_dump)

            if selection:
                self.data = {k: v for k, v in self.data.items() if selection in k}

        logger.info("Loaded %s experts from huggingface hub", len(self.data))

    def add_expert(self, expert_name: str, expert_dump: Expert):
        if expert_name in self.data:
            raise ValueError(f"Expert {expert_name} already exists")

        if expert_dump.expert_config.model != self.model_name:
            raise ValueError(
                f"Model {expert_dump.expert_config.model} does not match {self.model_name}"
            )

        self.data[expert_name] = expert_dump

    def add_expert_from_ckpt(self, ckpt_path: str, expert_name: str = None):
        expert_dump = load_expert(ckpt_path, expert_name)

        self.add_expert(expert_dump.expert_config.expert_name, expert_dump)

    def close(self):
        # synchronize with huggingface hub
        library_dump = {"expert_name": [], "expert_dump": []}
        for expert_name, expert_dump in self.data.items():
            library_dump["expert_name"].append(expert_name)
            library_dump["expert_dump"].append(expert_dump)

        with io.BytesIO() as buffer:
            torch.save(library_dump, buffer)
            buffer.flush()

            logger.info("Uploading library to huggingface hub...")
            logger.info("Num experts: %s", len(library_dump["expert_name"]))
            logger.info("Total size: %s MB", buffer.tell() / 1024 / 1024)

            addition = CommitOperationAdd(
                path_in_repo=f"library.ckpt", path_or_fileobj=buffer
            )
            preupload_lfs_files(self.repo_id, additions=[addition])
            create_commit(
                self.repo_id, operations=[addition], commit_message="Update library."
            )

        api = HfApi()

        def upload(buffer):
            api.upload_file(
                path_or_fileobj=buffer,
                path_in_repo="README.md",
                repo_id=self.repo_id,
            )

        with io.BytesIO() as buffer:
            # Write the following into the buffer:
            # This is a library of experts for the model {self.model_name}
            # Number of experts present in the library: {len(library_dump["expert_name"])}
            # Types of experts present in the library: unique(dump.model_modifier for dump in library_dump["expert_dump"])
            buffer.write(
                f"This is a library of experts for the model: {self.model_name}\n\n".encode(
                    "utf-8"
                )
            )
            buffer.write(
                f"Number of experts present in the library: {len(library_dump['expert_name'])}\n\n".encode(
                    "utf-8"
                )
            )
            buffer.write(
                f"| Expert Name | Trained on | Adapter Type |\n".encode("utf-8")
            )
            buffer.write(f"| --- | --- | --- |\n".encode("utf-8"))
            for expert_name, expert_dump in zip(
                library_dump["expert_name"], library_dump["expert_dump"]
            ):
                buffer.write(
                    f"| {expert_name} | {expert_dump.expert_config.dataset}/{expert_dump.expert_config.finetune_task_name} | {expert_dump.expert_config.model_modifier} |\n".encode(
                        "utf-8"
                    )
                )
            # write date before last updated on
            buffer.write(
                f"Last updated on: {api.repo_info(self.repo_id).last_modified}\n\n".encode(
                    "utf-8"
                )
            )
            buffer.flush()
            upload(buffer)


class LocalExpertLibrary(UserDict):
    def __init__(self, modules_dir, model_name, selection="", operator=np.argmin):
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

    def __getitem__(self, task):
        experts = self.data[task]
        if task == "base":
            return None
        if not isinstance(experts, list):
            return experts
        metrics = [
            float(e.split("/")[-1].replace(".ckpt", "").replace("loss=", ""))
            for e in experts
        ]
        args_best = self.operator(metrics)
        return experts[args_best]

    def pop(self, task, default=None):
        return self.data.pop(task, default)


if __name__ == "__main__":
    from mttl.utils import get_checkpoint_path, setup_logging

    setup_logging()

    library = HFExpertLibrary(
        "sordonia/test-library-for-neo-125m", "EleutherAI/gpt-neo-125m"
    )
    for directory in glob.glob("../../amlt/flan_experts_gptneo_125m_tleft/*"):
        library.add_expert_from_ckpt(
            get_checkpoint_path(directory),
        )
        break
    library.close()
