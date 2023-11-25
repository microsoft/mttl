from dataclasses import dataclass, asdict
import glob
import io
import json
import torch
import os
import numpy as np
from collections import UserDict

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub import (
    hf_hub_download,
    login,
    CommitOperationAdd,
    create_commit,
    snapshot_download,
    preupload_lfs_files,
    create_repo,
)

from huggingface_hub import HfApi
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.module_graph import (
    Expert,
    ExpertConfig,
    load_expert,
    ExpertInfo,
)


class ExpertLibrary:
    pass


@dataclass
class MetadataEntry:
    expert_name: str = None
    expert_info: ExpertInfo = None
    expert_config: ExpertConfig = None

    def dumps(self):
        return {
            "expert_config": self.expert_config.to_json(),
            "expert_info": asdict(self.expert_info),
            "expert_name": self.expert_name,
        }

    @classmethod
    def loads(cls, ckpt):
        return cls(
            expert_config=ExpertConfig(
                kwargs=json.loads(ckpt["expert_config"]),
                silent=True,
                raise_error=False,
            ),
            expert_info=ExpertInfo(**ckpt["expert_info"]),
            expert_name=ckpt["expert_name"],
        )


class HFExpertLibrary(ExpertLibrary):
    def __init__(self, repo_id, model_name=None, selection=None):
        super().__init__()

        self.repo_id = repo_id
        self.api = HfApi()
        self._sliced = False
        self.data = {}

        if "HF_TOKEN" in os.environ:
            login(token=os.environ["HF_TOKEN"])

        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
        except:
            pass

        metadata_dir = snapshot_download(repo_id, allow_patterns="*.meta")
        metadata = [
            MetadataEntry.loads(torch.load(file, map_location="cpu"))
            for file in glob.glob(f"{metadata_dir}/*.meta")
        ]

        for metadatum in metadata:
            if model_name is not None and metadatum.expert_config.model != model_name:
                self._sliced = True
                continue

            if metadatum.expert_name in self.data:
                raise ValueError(
                    f"Expert {metadata.expert_name} already exists. Library corrupted."
                )

            self.data[metadatum.expert_name] = metadatum

        if selection:
            self._sliced = True
            self.data = {k: v for k, v in self.data.items() if selection in k}

        logger.info("Loaded %s experts from huggingface hub", len(self.data))

    def _download_model(self, model_name):
        if model_name not in self.data:
            raise ValueError(f"Model {model_name} not found in repository.")

        model_file = f"{model_name}.ckpt"
        return hf_hub_download(self.repo_id, filename=model_file)

    def _upload_weights(self, expert_name, expert_dump):
        with io.BytesIO() as buffer:
            torch.save(expert_dump.expert_weights, buffer)
            buffer.flush()

            logger.info("Uploading expert to huggingface hub...")
            addition = CommitOperationAdd(
                path_in_repo=f"{expert_name}.ckpt", path_or_fileobj=buffer
            )
            preupload_lfs_files(self.repo_id, additions=[addition])
            create_commit(
                self.repo_id,
                operations=[addition],
                commit_message=f"Update library with {expert_name}.",
            )

    def _upload_metadata(self, metadata):
        with io.BytesIO() as buffer:
            torch.save(metadata.dumps(), buffer)
            buffer.flush()

            self.api.upload_file(
                repo_id=self.repo_id,
                path_or_fileobj=buffer,
                path_in_repo=f"{metadata.expert_name}.meta",
            )
            print(f"Metadata for {metadata.expert_name} uploaded successfully.")

    def keys(self):
        return list(self.data.keys())

    def items(self):
        for k in self.keys():
            yield k, self.__getitem__(k)

    def __getitem__(self, model_name):
        try:
            model = self._download_model(model_name)
            # Load the model from the downloaded file
            model = torch.load(model, map_location="cpu")
            return Expert(
                expert_config=self.data[model_name].expert_config,
                expert_info=self.data[model_name].expert_info,
                expert_weights=model,
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

    def __len__(self):
        return len(self.data)

    def add_expert(self, expert_name: str, expert_dump: Expert, force: bool = False):
        if self._sliced:
            raise ValueError("Cannot add expert to sliced library.")

        if expert_name in self.data and not force:
            raise ValueError(f"Expert {expert_name} already exists")

        metadata = MetadataEntry(
            expert_name=expert_name,
            expert_info=expert_dump.expert_info,
            expert_config=expert_dump.expert_config,
        )

        self._upload_weights(expert_name, expert_dump)
        self._upload_metadata(metadata)

        self.data[metadata.expert_name] = metadata
        self._update_readme()

    def _update_readme(self):
        api = HfApi()

        def upload(buffer):
            api.upload_file(
                path_or_fileobj=buffer,
                path_in_repo="README.md",
                repo_id=self.repo_id,
            )

        with io.BytesIO() as buffer:
            # Write the following into the buffer:
            # Number of experts present in the library: {len(library_dump["expert_name"])}
            # Types of experts present in the library: unique(dump.model_modifier for dump in library_dump["expert_dump"])
            buffer.write(
                f"Number of experts present in the library: {len(self)}\n\n".encode(
                    "utf-8"
                )
            )
            buffer.write(
                f"| Expert Name | Base Model | Trained on | Adapter Type |\n".encode(
                    "utf-8"
                )
            )
            buffer.write(f"| --- | --- | --- | --- |\n".encode("utf-8"))
            for expert_name, metadata in self.data.items():
                buffer.write(
                    f"| {expert_name} | {metadata.expert_config.model} | {metadata.expert_config.dataset}/{metadata.expert_config.finetune_task_name} | {metadata.expert_config.model_modifier} |\n".encode(
                        "utf-8"
                    )
                )

            # write date before last updated on
            buffer.write(
                f"Last updated on: {api.repo_info(self.repo_id).lastModified}\n\n".encode(
                    "utf-8"
                )
            )
            buffer.flush()
            upload(buffer)

    def add_expert_from_ckpt(
        self, ckpt_path: str, expert_name: str = None, force: bool = False
    ):
        expert_dump = load_expert(ckpt_path, expert_name)

        self.add_expert(expert_dump.expert_config.expert_name, expert_dump, force=force)


class LocalExpertLibrary(UserDict, ExpertLibrary):
    def __init__(self, modules_dir, model_name, selection="", operator=np.argmin):
        """
        Searches local experts
        """
        UserDict.__init__(self)

        self.home_dir = modules_dir
        # searches home and loads all the existing experts with selection criteria
        all_checkpoints = glob.glob(f"{self.home_dir}/**/*{selection}/*.ckpt")
        all_checkpoints += glob.glob(f"{self.home_dir}/**/**/*{selection}/*.ckpt")
        all_checkpoints += glob.glob(f"{self.home_dir}/**/{selection}.ckpt")
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
        if len(experts) == 1:
            return experts[0]
        metrics = [
            float(e.split("/")[-1].replace(".ckpt", "").replace("loss=", ""))
            for e in experts
        ]
        args_best = self.operator(metrics)
        return experts[args_best]

    def pop(self, task, default=None):
        return self.data.pop(task, default)
