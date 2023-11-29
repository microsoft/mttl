from contextlib import contextmanager
from dataclasses import dataclass
import glob
import io
import json
from typing import Any, Dict, List
import torch
import os
import numpy as np

from huggingface_hub import (
    hf_hub_download,
    login,
    CommitOperationAdd,
    CommitOperationDelete,
    CommitOperationCopy,
    create_commit,
    snapshot_download,
    preupload_lfs_files,
    create_repo,
    HfApi,
)
from huggingface_hub.utils._errors import RepositoryNotFoundError

from huggingface_hub import HfApi
from mttl.utils import logger
from mttl.models.modifiers.expert_containers.module_graph import (
    Expert,
    load_expert,
    ExpertInfo,
)


@dataclass
class MetadataEntry(ExpertInfo):
    expert_deleted: bool = False

    def dumps(self):
        dump = ExpertInfo.dumps(self)
        dump["expert_deleted"] = self.expert_deleted
        return dump

    @classmethod
    def loads(cls, ckpt):
        entry = cls(**ckpt)
        entry.expert_deleted = ckpt.get("expert_deleted", False)
        return entry


class BackendEngine:
    def snapshot_download(self, repo_id, allow_patterns=None):
        raise NotImplementedError

    def create_repo(self, repo_id, repo_type, exist_ok):
        raise NotImplementedError

    def create_commit(self, repo_id, operations, commit_message):
        raise NotImplementedError

    def preupload_lfs_files(self, repo_id, additions):
        raise NotImplementedError

    def hf_hub_download(self, repo_id, filename):
        raise NotImplementedError

    def login(self, token):
        raise NotImplementedError

    def repo_info(self, repo_id):
        raise NotImplementedError

    def list_repo_files(self, repo_id):
        raise NotImplementedError


class HuggingfaceHubEngine(BackendEngine):
    def snapshot_download(self, repo_id, allow_patterns=None):
        return snapshot_download(repo_id, allow_patterns=allow_patterns)

    def create_repo(self, repo_id, repo_type, exist_ok):
        return create_repo(repo_id, repo_type=repo_type, exist_ok=exist_ok)

    def create_commit(self, repo_id, operations, commit_message):
        return create_commit(
            repo_id, operations=operations, commit_message=commit_message
        )

    def preupload_lfs_files(self, repo_id, additions):
        return preupload_lfs_files(repo_id, additions=additions)

    def hf_hub_download(self, repo_id, filename):
        return hf_hub_download(repo_id, filename=filename)

    def repo_info(self, repo_id):
        return HfApi().repo_info(repo_id)

    def login(self, token):
        return login(token=token)

    def list_repo_files(self, repo_id):
        return HfApi().list_repo_files(repo_id)


class LocalFSEngine(BackendEngine):
    def snapshot_download(self, repo_id, allow_patterns=None):
        return repo_id

    def create_repo(self, repo_id, repo_type, exist_ok):
        os.makedirs(repo_id, exist_ok=exist_ok)

    def create_commit(self, repo_id, operations, commit_message):
        for op in operations:
            if type(op) == CommitOperationAdd:
                with open(os.path.join(repo_id, op.path_in_repo), "wb") as f:
                    f.write(op.path_or_fileobj.read())
            elif type(op) == CommitOperationCopy:
                import shutil

                shutil.copyfile(
                    os.path.join(repo_id, op.src_path_in_repo),
                    os.path.join(repo_id, op.path_in_repo),
                )
            elif type(op) == CommitOperationDelete:
                os.remove(os.path.join(repo_id, op.path_in_repo))

    def preupload_lfs_files(self, repo_id, additions):
        pass

    def hf_hub_download(self, repo_id, filename):
        return os.path.join(repo_id, filename)

    def login(self, token):
        pass

    def repo_info(self, repo_id):
        import datetime

        # return the current time into string format
        class RepoInfo:
            pass

        repo_info = RepoInfo()
        repo_info.lastModified = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return repo_info

    def list_repo_files(self, repo_id):
        import glob

        return list(glob.glob(os.path.join(repo_id, "*")))


class ExpertLibrary:
    def __init__(self, repo_id, model_name=None, selection=None, create=False):
        super().__init__()

        self.repo_id = repo_id
        self._sliced = False
        self.selection = selection
        self.model_name = model_name
        self._in_transaction = False
        self._pending_operations = []
        self._pending_pre_uploads = []
        self.data = {}

        if "HF_TOKEN" in os.environ:
            self.login(token=os.environ["HF_TOKEN"])

        try:
            if create:
                self.create_repo(repo_id, repo_type="model", exist_ok=True)
        except Exception as e:
            logger.error("Error creating repo %s\n", repo_id)

        self._build_lib()
        logger.info("Loaded %s experts from huggingface hub", len(self.data))

    def _build_lib(self):
        self._sliced = False
        self.data = {}

        try:
            metadata_dir = self.snapshot_download(self.repo_id, allow_patterns="*.meta")
        except Exception as e:
            if isinstance(e, RepositoryNotFoundError):
                logger.error("Repository not found: %s", self.repo_id)
                return self.data, self._sliced
            raise e

        metadata = [
            MetadataEntry.loads(torch.load(file, map_location="cpu"))
            for file in glob.glob(f"{metadata_dir}/*.meta")
        ]

        for metadatum in metadata:
            if self.model_name is not None and metadatum.model != self.model_name:
                self._sliced = True
                continue
            if metadatum.expert_deleted:
                continue

            key = metadatum.expert_name
            if key in self.data:
                raise ValueError(
                    f"Expert {metadatum.expert_name} already exists. Library corrupted."
                )
            self.data[key] = metadatum

        if self.selection:
            self._sliced = True
            self.data = {k: v for k, v in self.data.items() if self.selection in k}

    def _download_model(self, model_name):
        if model_name not in self.data:
            raise ValueError(f"Model {model_name} not found in repository.")

        model_file = f"{model_name}.ckpt"
        return self.hf_hub_download(self.repo_id, filename=model_file)

    def _upload_weights(self, expert_name, expert_dump):
        buffer = io.BytesIO()
        torch.save(expert_dump.expert_weights, buffer)
        buffer.flush()

        logger.info("Uploading expert to huggingface hub...")
        addition = CommitOperationAdd(
            path_in_repo=f"{expert_name}.ckpt", path_or_fileobj=buffer
        )
        if self._in_transaction:
            self._pending_pre_uploads.append(addition)
            self._pending_operations.append(addition)
        else:
            self.preupload_lfs_files(self.repo_id, additions=[addition])
            self.create_commit(
                self.repo_id,
                operations=[addition],
                commit_message=f"Update library with {expert_name}.",
            )
            logger.info(f"Expert {expert_name} uploaded successfully.")

    def _upload_metadata(self, metadata):
        buffer = io.BytesIO()
        torch.save(metadata.__dict__, buffer)
        buffer.flush()

        addition = CommitOperationAdd(
            path_in_repo=f"{metadata.expert_name}.meta", path_or_fileobj=buffer
        )

        if self._in_transaction:
            self._pending_operations.append(addition)
        else:
            self.create_commit(
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

    def get_expert(self, expert_name, with_auxiliary_data: bool = False):
        expert_dump = self[expert_name]

        if with_auxiliary_data:
            embeddings = self.get_auxiliary_data(
                data_type="embeddings", expert_name=expert_name
            )
            scores = self.get_auxiliary_data(
                data_type="scores", expert_name=expert_name
            )
            expert_dump.expert_info.embeddings = embeddings
            expert_dump.expert_info.scores = scores
        return expert_dump

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
            expert_info=self.data[model_name],
            expert_weights=model,
        )

    def __len__(self):
        return len(self.data)

    def add_expert(
        self, expert_dump: Expert, expert_name: str = None, force: bool = False
    ):
        if self._sliced:
            raise ValueError("Cannot add expert to sliced library.")

        if expert_name is not None:
            expert_dump.expert_info.expert_name = expert_name

        if expert_dump.expert_info.expert_name in self.data and not force:
            raise ValueError(f"Expert {expert_name} already exists!")

        if "." in expert_dump.expert_info.expert_name:
            raise ValueError("Expert name cannot contain dots.")

        # convert to metadata entry
        metadata = MetadataEntry.loads(expert_dump.expert_info.__dict__)

        self._upload_weights(metadata.expert_name, expert_dump)
        self._upload_metadata(metadata)

        self.data[metadata.expert_name] = metadata
        self._update_readme()

    def get_auxiliary_data(
        self,
        data_type: str = "embeddings",
        expert_name: str = None,
    ) -> List[Any]:
        path = snapshot_download(self.repo_id, allow_patterns=f"*.{data_type}")

        if expert_name:
            filename = os.path.join(path, f"{expert_name}.{data_type}")
            if not os.path.isfile(filename):
                raise ValueError(
                    f"Data of type {data_type} for expert {expert_name} not found in repository. Did you compute it?"
                )
            return torch.load(filename)
        else:
            auxiliary_data = {}
            for key in self.keys():
                filename = os.path.join(path, f"{key}.{data_type}")
                if os.path.isfile(filename):
                    auxiliary_data[f"{key}"] = torch.load(filename)
        return auxiliary_data

    def unremove_expert(self, expert_name: str):
        """Restore a previously soft-deleted expert."""
        if self._sliced:
            raise ValueError("Cannot remove expert from sliced library.")

        list_of_files = self.list_repo_files(self.repo_id)
        if f"{expert_name}.meta" not in list_of_files:
            raise ValueError(f"Expert {expert_name} not found in repository.")

        path = self.hf_hub_download(self.repo_id, filename=f"{expert_name}.meta")
        metadata = MetadataEntry.loads(torch.load(path, map_location="cpu"))
        metadata.expert_deleted = False

        self._upload_metadata(metadata)
        self.data[expert_name] = metadata

    def remove_expert(self, expert_name: str, soft_delete: bool = True):
        """Remove an expert from the library.

        soft_delete: if True, the expert is not removed from the repository, but only marked as deleted.
        """
        if self._sliced:
            raise ValueError("Cannot remove expert from sliced library.")

        if expert_name not in self.data:
            raise ValueError(f"Expert {expert_name} not found in repository.")

        if not soft_delete:
            deletion_a = CommitOperationDelete(path_in_repo=f"{expert_name}.ckpt")
            deletion_b = CommitOperationDelete(path_in_repo=f"{expert_name}.meta")

            if self._in_transaction:
                # watch out, if other operations (adding files) are pending, this might be dangerous
                self._pending_operations.extend([deletion_a, deletion_b])
            else:
                self.create_commit(
                    self.repo_id,
                    operations=[deletion_a, deletion_b],
                    commit_message=f"Update library with {expert_name}.",
                )
                logger.info(f"Deletion of {expert_name} successful.")
        else:
            metadata = self.data[expert_name]
            metadata.expert_deleted = True
            self._upload_metadata(metadata)

        self.data.pop(expert_name)
        self._update_readme()

    def add_scores(
        self,
        expert_name: str,
        scores_config: Dict,
        expert_scores: np.ndarray,
    ):
        if expert_name not in self.data:
            raise ValueError(f"Expert {expert_name} not found in repository.")

        if "name" not in scores_config:
            raise ValueError("Embedding config must contain a name.")

        operations = []
        scores_file = f"{expert_name}.scores"

        scores = self.list_repo_files(self.repo_id)
        if scores_file in scores:
            path = self.hf_hub_download(self.repo_id, filename=scores_file)
            scores = torch.load(path, map_location="cpu")
        else:
            scores = {}

        scores[scores_config["name"]] = {
            "scores": expert_scores,
            "config": scores_config,
        }

        buffer = io.BytesIO()
        torch.save(buffer, scores)
        buffer.flush()

        addition_a = CommitOperationAdd(
            path_in_repo=f"{scores_file}", path_or_fileobj=buffer
        )
        operations.append(addition_a)

        if self._in_transaction:
            self._pending_operations.extend(operations)
        else:
            self.create_commit(
                self.repo_id,
                operations=operations,
                commit_message=f"Update library with embedding for {expert_name}.",
            )
            logger.info(f"Scores for {expert_name} uploaded successfully.")

    def add_embeddings(
        self,
        expert_name: str,
        embedding_config: Dict,
        expert_embedding: np.ndarray,
    ):
        if expert_name not in self.data:
            raise ValueError(f"Expert {expert_name} not found in repository.")

        if "name" not in embedding_config:
            raise ValueError("Embedding config must contain a name.")

        operations = []
        embedding_file = f"{expert_name}.embeddings"

        embeddings = self.list_repo_files(self.repo_id)
        if embedding_file in embeddings:
            path = self.hf_hub_download(self.repo_id, filename=embedding_file)
            embeddings = torch.load(path, map_location="cpu")
        else:
            embeddings = {}

        embeddings[embedding_config["name"]] = {
            "embedding": expert_embedding,
            "config": embedding_config,
        }

        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.flush()

        addition_a = CommitOperationAdd(
            path_in_repo=f"{embedding_file}", path_or_fileobj=buffer
        )
        operations.append(addition_a)

        if self._in_transaction:
            self._pending_operations.extend(operations)
        else:
            self.create_commit(
                self.repo_id,
                operations=operations,
                commit_message=f"Update library with embedding for {expert_name}.",
            )
            logger.info(f"Embedding for {expert_name} uploaded successfully.")

    def _update_readme(self):
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
                f"| {expert_name} | {metadata.model} | {metadata.dataset}/{metadata.expert_task_name} | {metadata.model_modifier} |\n".encode(
                    "utf-8"
                )
            )

        # write date before last updated on
        buffer.write(
            f"Last updated on: {self.repo_info(self.repo_id).lastModified}\n\n".encode(
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
            self.create_commit(
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
        if self._pending_pre_uploads:
            preupload_lfs_files(self.repo_id, additions=self._pending_pre_uploads)
        self.create_commit(
            self.repo_id,
            operations=self._pending_operations,
            commit_message="Update library with new ops.",
        )
        # exit transaction and clear pending operations
        self._in_transaction = False
        self._pending_pre_uploads.clear()
        self._pending_operations.clear()

    def add_expert_from_ckpt(
        self, ckpt_path: str, expert_name: str = None, force: bool = False
    ):
        expert_dump = load_expert(ckpt_path)
        self.add_expert(expert_dump, expert_name=expert_name, force=force)

    def rename_expert(self, old_name, new_name):
        if self._sliced:
            raise ValueError("Cannot rename expert in sliced library.")

        if old_name not in self.data:
            raise ValueError(f"Expert {old_name} not found in repository.")

        if new_name in self.data:
            raise ValueError(f"Expert {new_name} already exists.")

        metadata = self.data[old_name]
        metadata.expert_name = new_name
        metadata.expert_config.expert_name = new_name

        self.data[new_name] = metadata
        self.data.pop(old_name)

        meta_delete = CommitOperationDelete(path_in_repo=f"{old_name}.meta")
        ckpt_copy = CommitOperationCopy(
            src_path_in_repo=f"{old_name}.ckpt", path_in_repo=f"{new_name}.ckpt"
        )
        ckpt_delete = CommitOperationDelete(path_in_repo=f"{old_name}.ckpt")
        ops = [meta_delete, ckpt_copy, ckpt_delete]

        if self._in_transaction:
            self._pending_operations.extend(ops)
        else:
            self.create_commit(
                self.repo_id,
                operations=ops,
                commit_message=f"Renaming expert {old_name} with {new_name}.",
            )
            logger.info(f"Expert {new_name} uploaded successfully.")

        self._upload_metadata(metadata)
        self._update_readme()

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


class HFExpertLibrary(ExpertLibrary, HuggingfaceHubEngine):
    pass


class LocalExpertLibrary(ExpertLibrary, LocalFSEngine):
    pass
