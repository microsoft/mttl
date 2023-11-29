from contextlib import contextmanager
from dataclasses import dataclass, asdict
import glob
import io
import json
from typing import Any
import torch
import os
import numpy as np
from collections import UserDict

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
class LibraryEmbedding:
    # names of the experts
    expert_names: list
    # embeddings of the experts
    expert_embeddings: np.ndarray
    # how the embeddings were computed
    config: Any = None


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
    def __init__(self, repo_id, model_name=None, selection=None, create=False):
        super().__init__()

        self.repo_id = repo_id
        self.api = HfApi()
        self._sliced = False
        self._in_transaction = False
        self._pending_operations = []
        self.data = {}

        if "HF_TOKEN" in os.environ:
            login(token=os.environ["HF_TOKEN"])

        try:
            if create:
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
        buffer = io.BytesIO()
        torch.save(expert_dump.expert_weights, buffer)
        buffer.flush()

        logger.info("Uploading expert to huggingface hub...")
        addition = CommitOperationAdd(
            path_in_repo=f"{expert_name}.ckpt", path_or_fileobj=buffer
        )
        preupload_lfs_files(self.repo_id, additions=[addition])
        if self._in_transaction:
            self._pending_operations.append(addition)
        else:
            create_commit(
                self.repo_id,
                operations=[addition],
                commit_message=f"Update library with {expert_name}.",
            )
            logger.info(f"Expert {expert_name} uploaded successfully.")

    def _upload_metadata(self, metadata):
        buffer = io.BytesIO()
        torch.save(metadata.dumps(), buffer)
        buffer.flush()

        addition = CommitOperationAdd(
            path_in_repo=f"{metadata.expert_name}.meta", path_or_fileobj=buffer
        )

        if self._in_transaction:
            self._pending_operations.append(addition)
        else:
            create_commit(
                self.repo_id,
                operations=[addition],
                commit_message=f"Update library with {metadata.expert_name}.",
            )
            logger.info(f"Metadata for {metadata.expert_name} uploaded successfully.")

    def keys(self):
        return self.data.keys()

    def items(self):
        for k in list(self.keys()):
            yield k, self.__getitem__(k)

    def __getitem__(self, model_name):
        if self._in_transaction:
            raise ValueError(
                "Cannot access library while in transaction. Finish current commit!"
            )

        if model_name not in self.data:
            raise ValueError(f"Expert {model_name} not found in repository.")

        model = self._download_model(model_name)
        # Load the model from the downloaded file
        model = torch.load(model, map_location="cpu")
        return Expert(
            expert_config=self.data[model_name].expert_config,
            expert_info=self.data[model_name].expert_info,
            expert_weights=model,
        )

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

    def read_embeddings(
        self,
        embedding_type: str,
    ) -> LibraryEmbedding:
        files = self.api.list_repo_files(self.repo_id)
        embedding_file = f"{embedding_type}.emb"
        config_file = f"{embedding_type}.json"

        if embedding_file not in files:
            raise ValueError(
                f"Embedding {embedding_file} not found in repository. Did you compute it?"
            )

        path = hf_hub_download(self.repo_id, filename=embedding_file)
        embeddings = torch.load(path, map_location="cpu")

        config = None
        if config_file in files:
            path = hf_hub_download(self.repo_id, filename=config_file)
            with open(path, "r") as path:
                config = json.load(path)

        return LibraryEmbedding(
            expert_names=embeddings["expert_names"],
            expert_embeddings=embeddings["expert_embeddings"],
            config=config,
        )

    def add_embeddings(
        self,
        embedding_type: str,
        expert_names: str,
        expert_embeddings: np.ndarray,
        config: Any = None,
        overwrite: bool = False,
    ):
        import json

        operations = []
        embedding_file = f"{embedding_type}.emb"
        config_file = f"{embedding_type}.json"

        embeddings = self.api.list_repo_files(self.repo_id)
        if embedding_file in embeddings:
            raise ValueError(
                f"Embedding {embedding_file} already exists. Use `overwrite=True`."
            )

        for expert_name in expert_names:
            if expert_name not in self.data:
                raise ValueError(f"Expert {expert_name} not found in repository.")

        buffer = io.BytesIO()
        torch.save(
            {"expert_names": expert_names, "expert_embeddings": expert_embeddings},
            buffer,
        )
        buffer.flush()

        addition_a = CommitOperationAdd(
            path_in_repo=f"{embedding_file}", path_or_fileobj=buffer
        )
        operations.append(addition_a)

        if config is not None:
            buffer = io.BytesIO()
            buffer.write(json.dumps(config.__dict__).encode("utf-8"))
            buffer.flush()

            addition_b = CommitOperationAdd(
                path_in_repo=f"{config_file}", path_or_fileobj=buffer
            )
            operations.append(addition_b)

        if self._in_transaction:
            self._pending_operations.extend(operations)
        else:
            create_commit(
                self.repo_id,
                operations=operations,
                commit_message=f"Update library with embedding for {expert_name}.",
            )
            logger.info(f"Embedding for {expert_name} uploaded successfully.")

    def _update_readme(self):
        api = HfApi()

        buffer = io.BytesIO()
        buffer.write(
            f"Number of experts present in the library: {len(self)}\n\n".encode("utf-8")
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

        addition = CommitOperationAdd(path_in_repo=f"README.md", path_or_fileobj=buffer)
        if self._in_transaction:
            # remove previous readme operations, keep only the latest
            for operation in self._pending_operations:
                if operation.path_in_repo == "README.md":
                    self._pending_operations.remove(operation)
            self._pending_operations.append(addition)
        else:
            create_commit(
                self.repo_id,
                operations=[addition],
                commit_message="Update readme.",
            )

    @contextmanager
    def batched_commit(self):
        """Context manager batching operations into a single commit."""
        # set in transaction flag
        self._in_transaction = True
        yield
        logger.info(f"Committing len(self._pending_operations) operations...")
        create_commit(
            self.repo_id,
            operations=self._pending_operations,
            commit_message="Update library with new ops.",
        )
        # exit transaction and clear pending operations
        self._in_transaction = False
        self._pending_operations.clear()

    def _commit(self):
        create_commit(
            self.repo_id,
            operations=self._pending_operations,
            commit_message="Update library with new experts.",
        )
        self._pending_operations = []

    def add_expert_from_ckpt(
        self, ckpt_path: str, expert_name: str = None, force: bool = False
    ):
        expert_dump = load_expert(ckpt_path, expert_name)

        self.add_expert(expert_dump.expert_config.expert_name, expert_dump, force=force)

    @property
    def tasks(self):
        """
        Assume that the experts' names correspond to the tasks they were trained on
        """
        return list(self.keys())

    def filter_with_tasks(self, tasks):
        """
        Remove modules for tasks other than the ones in tasks.
        """
        self._sliced = True
        all_tasks = self.tasks

        for t in all_tasks:
            if t not in tasks:
                self.data.pop(t, None)


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
