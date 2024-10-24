import glob
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import total_ordering
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)
from huggingface_hub.errors import RepositoryNotFoundError

from mttl.logging import logger
from mttl.models.library.backend_engine import (
    BlobStorageEngine,
    HuggingfaceHubEngine,
    LocalFSEngine,
)
from mttl.models.library.expert import Expert, ExpertInfo, load_expert


@total_ordering
@dataclass
class Score:
    name: str
    task: str
    split: str
    value: np.ndarray = None
    config: Dict[str, Any] = None

    @property
    def key(self):
        return (self.name, self.task, self.split)

    @property
    def hash(self) -> str:
        return str(self.key).encode()

    @classmethod
    def fromdict(self, data):
        return Score(**data)

    def asdict(self):
        return self.__dict__

    def __lt__(self, other):
        if not isinstance(other, Score):
            return self.value < other
        return self.value < other.value

    def __eq__(self, other):
        if not isinstance(other, Score):
            return self.value == other
        return self.value == other.value


@dataclass
class MetadataEntry(ExpertInfo):
    expert_deleted: bool = False


class ExpertLibrary:
    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        model_name: Optional[str] = None,
        selection: Optional[List[str]] = None,
        exclude_selection: Optional[List[str]] = None,
        create: bool = False,
        ignore_sliced: bool = False,
    ):
        super().__init__()

        self.protocol, self.repo_id = self._remove_protocol(repo_id)
        self._uri = repo_id
        self._sliced = False
        self.selection = selection
        self.exclude_selection = exclude_selection
        self.model_name = model_name
        self._in_transaction = False
        self._pending_operations = []
        self._pending_pre_uploads = []
        self.data = {}

        self.ignore_sliced = ignore_sliced

        if self.selection and self.exclude_selection:
            raise ValueError("Cannot use both selection and exclude_selection.")

        self.login(token=token)

        try:
            if create:
                self.create_repo(
                    self.repo_id, repo_type="model", exist_ok=True, private=True
                )
        except Exception as e:
            logger.error("Error creating repo %s.", repo_id)
            logger.error(e)
            sys.exit(1)

        self._build_lib()
        logger.info("Loaded %s experts from remote storage", len(self.data))

    @property
    def uri(self):
        return self._uri

    @staticmethod
    def _remove_protocol(repo_id) -> Tuple[str, str]:
        """Remove the protocol from the repo_id. Ex:
        az://storage_account/container -> storage_account/container
        """
        paths = str(repo_id).split("://")
        return "hf" if len(paths) == 1 else paths[0], paths[-1]

    @property
    def sliced(self):
        return self._sliced and not self.ignore_sliced

    def _build_lib(self):
        self._sliced = False
        self.data = {}

        try:
            metadata_dir = self.snapshot_download(
                self.repo_id, allow_patterns=["**/*.meta", "*.meta"]
            )
        except Exception as e:
            if isinstance(e, RepositoryNotFoundError):
                logger.error("Repository not found: %s", self.repo_id)
            raise e

        metadata = [
            MetadataEntry.fromdict(
                torch.load(file, map_location="cpu", weights_only=False)
            )
            for file in glob.glob(f"{metadata_dir}/**/*.meta", recursive=True)
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
            logger.warning("Only including experts in selection: %s", self.selection)
            self._sliced = True
            self.data = {k: v for k, v in self.data.items() if k in self.selection}
        elif self.exclude_selection:
            logger.warning("Excluding experts in selection: %s", self.exclude_selection)
            self._sliced = True
            self.data = {
                k: v for k, v in self.data.items() if k not in self.exclude_selection
            }

    def refresh_from_remote(self):
        self._build_lib()

    def _download_model(self, model_name):
        if model_name not in self.data:
            raise ValueError(f"Model {model_name} not found in repository.")

        model_file = f"{model_name}.ckpt"
        return self.hf_hub_download(self.repo_id, filename=model_file)

    def _upload_weights(self, expert_name, expert_dump):
        buffer = io.BytesIO()
        torch.save(expert_dump.expert_weights, buffer)
        buffer.flush()

        logger.info(f"Uploading expert to {self.repo_id}...")
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
        torch.save(metadata.asdict(), buffer)
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
            scores = self.get_auxiliary_data(
                data_type="scores", expert_name=expert_name
            )
            expert_dump.expert_info.scores = scores
        return expert_dump

    def __getitem__(self, expert_name):
        if self._in_transaction:
            raise ValueError(
                "Cannot access library while in transaction. Finish current commit!"
            )

        if expert_name not in self.data:
            raise ValueError(f"Expert {expert_name} not found in repository.")

        model = self._download_model(expert_name)
        # Load the model from the downloaded file
        model = torch.load(model, map_location="cpu", weights_only=True)

        return Expert(
            expert_info=self.data[expert_name],
            expert_weights=model,
        )

    def __len__(self):
        return len(self.data)

    def add_expert(
        self, expert_dump: Expert, expert_name: str = None, force: bool = False
    ):
        if self.sliced:
            raise ValueError("Cannot add expert to sliced library.")

        if expert_name is not None:
            # why would we want to do it?
            expert_dump.expert_info = replace(
                expert_dump.expert_info, expert_name=expert_name
            )

        if expert_dump.expert_info.expert_name in self.data and not force:
            raise ValueError(
                f"Expert {expert_dump.expert_info.expert_name} already exists!"
            )

        if "." in expert_dump.expert_info.expert_name:
            raise ValueError("Expert name cannot contain dots.")

        # convert to metadata entry
        metadata = MetadataEntry.fromdict(expert_dump.expert_info.asdict())

        self._upload_weights(metadata.expert_name, expert_dump)
        self._upload_metadata(metadata)
        self.data[metadata.expert_name] = metadata
        self._update_readme()

    def list_auxiliary_data(self) -> Dict[str, Tuple[int, str]]:
        """List auxiliary data in the library, returns a dictionary with the data type, the number of records, and a string representation of the config file."""
        auxiliary_data = {}
        list_of_files = [
            f for f in self.list_repo_files(self.repo_id) if f.endswith(f".bin")
        ]
        for file in list_of_files:
            try:
                _, data_type, _ = os.path.basename(file).split(".")
            except:
                continue
            if data_type in auxiliary_data:
                auxiliary_data[data_type][0] += 1
            else:
                path = self.hf_hub_download(self.repo_id, filename=file)
                try:
                    config = repr(torch.load(path, weights_only=False)["config"])
                except:
                    # old format
                    config = "N/A"
                auxiliary_data[data_type] = [1, config]
        return auxiliary_data

    def get_auxiliary_data(
        self,
        data_type: str,
        expert_name: str = None,
        return_config: bool = False,
    ) -> List[Any]:
        """Get auxiliary data from the repository.

        Args:
            data_type (str): data type (basically the filename in the lib repo)
            expert_name (str, optional): expert name to fetch auxiliary data for. If None, fetches for every expert.
            return_config (bool, optional): if True, returns {"config": config, "data": data}, else it returns only the raw data.

        Raises:
            ValueError: _description_

        Returns:
            List[Any]: _description_
        """
        # all auxiliary data should have "bin" extension
        path = self.snapshot_download(self.repo_id, allow_patterns=f"*.{data_type}.bin")

        if expert_name:
            filename = os.path.join(path, f"{expert_name}.{data_type}.bin")
            if not os.path.isfile(filename):
                raise ValueError(
                    f"Data of type {data_type} for expert {expert_name} not found in repository. Did you compute it?"
                )
            payload = torch.load(filename, weights_only=False)
            if return_config and "config" in payload:
                return payload["config"], payload["data"]
            if "data" in payload:
                return payload["data"]
            return payload
        else:
            auxiliary_data = {}
            for key in self.keys():
                filename = os.path.join(path, f"{key}.{data_type}.bin")
                if os.path.isfile(filename):
                    payload = torch.load(filename, weights_only=False)
                    if return_config and "config" in payload:
                        auxiliary_data[f"{key}"] = payload["config"], payload["data"]
                    elif "data" in payload:
                        auxiliary_data[f"{key}"] = payload["data"]
                    else:
                        auxiliary_data[f"{key}"] = payload
        return auxiliary_data

    def remove_auxiliary_data(
        self,
        data_type: str = None,
        expert_name: str = None,
    ):
        """Remove auxiliary data from the repository."""
        if self.sliced:
            raise ValueError("Cannot remove auxiliary data from sliced library.")

        # all aux data ends with .bin
        list_of_files = [
            f for f in self.list_repo_files(self.repo_id) if f.endswith(".bin")
        ]
        files_to_remove = []

        if (
            expert_name
            and data_type
            and f"{expert_name}.{data_type}.bin" not in list_of_files
        ):
            raise ValueError(
                f"Auxiliary data of type {data_type} for expert {expert_name} not found in repository."
            )

        if expert_name is None and data_type is None:
            # remove all auxiliary data from the library
            files_to_remove = list_of_files
        elif expert_name is not None and data_type is None:
            files_to_remove = [
                f for f in list_of_files if f.startswith(f"{expert_name}.")
            ]
        else:
            files_to_remove = [f"{expert_name}.{data_type}.bin"]

        deletion_ops = []
        for file in files_to_remove:
            deletion = CommitOperationDelete(path_in_repo=file)
            deletion_ops.append(deletion)

        if self._in_transaction:
            self._pending_operations.extend(deletion_ops)
        else:
            self.create_commit(
                self.repo_id,
                operations=deletion_ops,
                commit_message=f"Deleting auxiliary data from the library.",
            )
            logger.info(f"Deletion of {files_to_remove} successful.")

    def unremove_expert(self, expert_name: str):
        """Restore a previously soft-deleted expert."""
        if self.sliced:
            raise ValueError("Cannot remove expert from sliced library.")

        list_of_files = self.list_repo_files(self.repo_id)
        if f"{expert_name}.meta" not in list_of_files:
            raise ValueError(f"Expert {expert_name} not found in repository.")

        path = self.hf_hub_download(self.repo_id, filename=f"{expert_name}.meta")
        metadata = MetadataEntry.fromdict(
            torch.load(path, weights_only=False, map_location="cpu")
        )
        metadata.expert_deleted = False

        self._upload_metadata(metadata)
        self.data[expert_name] = metadata

    def remove_expert(self, expert_name: str, soft_delete: bool = True):
        """Remove an expert from the library.

        soft_delete: if True, the expert is not removed from the repository, but only marked as deleted.
        """
        if self.sliced:
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

        metadata = self.data.pop(expert_name)
        self._update_readme()

    def get_score(self, expert_name: str, hash: str):
        try:
            scores = self.get_auxiliary_data(
                data_type="scores", expert_name=expert_name
            )
        except ValueError:
            return None
        if hash not in scores:
            return None
        return Score(**scores[hash])

    def add_score(self, expert_name: str, score: Score):
        if expert_name not in self.data:
            raise ValueError(f"Expert {expert_name} not found in repository.")

        operations = []
        scores_file = f"{expert_name}.scores.bin"

        scores = self.list_repo_files(self.repo_id)
        if scores_file in scores:
            path = self.hf_hub_download(self.repo_id, filename=scores_file)
            scores = torch.load(path, weights_only=False, map_location="cpu")
        else:
            scores = {}

        task = score.task
        if score.hash in scores:
            raise ValueError(f"Score {score.name} already exists for task {task}.")
        if score.value is None:
            raise ValueError(f"Score {score.name} has no value and cannot be added.")
        scores[score.hash] = score.asdict()

        buffer = io.BytesIO()
        torch.save(scores, buffer)
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

    def add_auxiliary_data(
        self,
        data_type: str,
        expert_name: str,
        config: Dict,
        data: Any,
        force: bool = False,
    ):
        if expert_name not in self.data and expert_name != "base_model":
            raise ValueError(f"Expert {expert_name} not found in repository.")

        operations = []
        aux_file = f"{expert_name}.{data_type}.bin"

        aux_data = self.list_repo_files(self.repo_id)
        if aux_file in aux_data and not force:
            raise ValueError(
                f"Data of type {data_type} for expert {expert_name} already exists in repository. Set `force=True` to overwrite."
            )

        payload = {
            "data": data,
            "config": config,
        }

        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.flush()

        addition_a = CommitOperationAdd(
            path_in_repo=f"{aux_file}", path_or_fileobj=buffer
        )
        operations.append(addition_a)

        if self._in_transaction:
            self._pending_pre_uploads.extend(operations)
            self._pending_operations.extend(operations)
        else:
            self.preupload_lfs_files(self.repo_id, additions=operations)
            self.create_commit(
                self.repo_id,
                operations=operations,
                commit_message=f"Upload auxiliary data {data_type} for {expert_name}.",
            )
            logger.info(
                f"Auxiliary data {data_type} for {expert_name} uploaded successfully."
            )

    def add_embeddings(
        self,
        expert_name: str,
        embedding_config: Dict,
        expert_embedding: np.ndarray,
        force: bool = False,
    ):
        return self.add_auxiliary_data(
            data_type="embeddings",
            expert_name=expert_name,
            config=embedding_config,
            data=expert_embedding,
            force=force,
        )

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
                f"| {expert_name} | {metadata.model} | {metadata.dataset}/{metadata.expert_task_name} | {metadata.modifier_name} |\n".encode(
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
        if len(self._pending_operations) == 0:
            self._in_transaction = False
            return
        logger.info(f"Committing {len(self._pending_operations)} operations...")

        if self._pending_pre_uploads:
            self.preupload_lfs_files(self.repo_id, additions=self._pending_pre_uploads)
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
        expert_dump = load_expert(ckpt_path, expert_name=expert_name)
        if expert_dump.name is None:
            raise ValueError(
                "Expert name not found in checkpoint. Need to explicitly provide one as argument."
            )

        self.add_expert(expert_dump, force=force)

    def rename_expert(self, old_name, new_name):
        if self.sliced:
            raise ValueError("Cannot rename expert in sliced library.")

        if old_name not in self.data:
            raise ValueError(f"Expert {old_name} not found in repository.")

        if new_name in self.data:
            raise ValueError(f"Expert {new_name} already exists.")

        metadata = self.data[old_name]
        metadata.expert_name = new_name

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
        Doesn't assume that the experts' names correspond to the tasks they were trained on
        """
        tasks = set()
        for metadatum in self.data.values():
            tasks.add(metadatum.expert_task_name)
        return list(tasks)

    def __contains__(self, expert: Union[Expert, str]):
        key = expert if isinstance(expert, str) else expert.expert_info.expert_name
        return key in self.data

    def replace_expert(self, old_expert: Expert, new_expert: Expert, soft_delete=True):
        """
        Replace an expert with a new one.
        """
        if old_expert in self and old_expert is not None:
            self.remove_expert(old_expert.name, soft_delete=soft_delete)
        return self.add_expert(new_expert)

    def get_experts_for_task(self, task):
        return [
            metadatum
            for metadatum in self.data.values()
            if metadatum.expert_task_name == task
        ]

    @classmethod
    def from_expert_library(
        cls,
        expert_lib: "ExpertLibrary",
        repo_id,
        force=False,
        upload_aux_data=False,
        only_tasks=None,
    ):
        new_lib = cls(repo_id=repo_id, create=True)

        only_tasks = only_tasks or expert_lib.tasks
        with new_lib.batched_commit():
            for name, expert in expert_lib.items():
                if expert.name not in new_lib:
                    new_lib.add_expert(expert, name, force=force)

        # if the new_lib already exists, delete experts that
        # are in this lib but were deleted from the expert_lib
        with new_lib.batched_commit():
            for name, metadatum in list(new_lib.data.items()):
                if (
                    name not in expert_lib.keys()
                    and metadatum.expert_task_name in only_tasks
                ):
                    new_lib.remove_expert(name, soft_delete=True)

        # also update the scores
        if upload_aux_data:
            scores = expert_lib.get_auxiliary_data(data_type="scores")
            for expert_name, expert_scores in scores.items():
                for score in expert_scores.values():
                    try:
                        new_lib.add_score(expert_name, Score(**score))
                    except ValueError as e:
                        logger.error(e)
                        continue

            # TODO: upload the embeddings
            embeddings = expert_lib.get_auxiliary_data(data_type="embeddings")
            for expert_name, expert_embeddings in embeddings.items():
                for embedding in expert_embeddings.values():
                    try:
                        new_lib.add_embeddings(
                            expert_name, embedding["config"], embedding["embeddings"]
                        )
                    except ValueError as e:
                        logger.error(e)
                        continue
        return new_lib

    def update_from_expert_library(self, expert_library: Union["ExpertLibrary", str]):
        """
        Update the expert library with experts from an existing ExpertLibrary.

        Args:
            expert_library (Union[ExpertLibrary, str]): The ExpertLibrary to update from.
            It can be either an instance of `ExpertLibrary` or a string representing the expert library.
        """
        if isinstance(expert_library, str):
            expert_library = self.get_expert_library(expert_library)
        for name, expert in expert_library.items():
            if expert not in self and not expert.expert_info.expert_deleted:
                self.add_expert(expert)
            if expert.expert_info.expert_deleted:
                self.remove_expert(expert.name, soft_delete=True)

    @classmethod
    def from_expert_dict(
        cls,
        expert_dict: Dict[str, Expert],
        destination: str,
        expert_library_type: Optional[Union["ExpertLibrary", str]] = None,
        create: bool = False,
        force: bool = False,
    ):
        """
        Create a new ExpertLibrary object from a dictionary of experts.
        Useful e.g. when I want to create a library from new experts that are created dynamically.

        Args:
            expert_dict (Dict[str, Expert]): A dictionary containing expert names as keys and Expert objects as values.
            destination (str): path where the library will be stored.

        Returns:
            ExpertLibrary: A new ExpertLibrary object containing the experts from the dictionary.
        """
        new_lib = cls.get_expert_library(
            repo_id=destination,
            expert_library_type=expert_library_type or cls,
            create=create,
        )
        with new_lib.batched_commit():
            for _, expert in expert_dict.items():
                if expert.name not in new_lib or force:
                    new_lib.add_expert(expert, force=force)
        return new_lib

    @staticmethod
    def _get_expert_lib_class(repo_id, expert_library_type):
        """Decide which ExpertLibrary subclass to use based on the repo_id."""
        available_libraries = {
            "local": LocalExpertLibrary,
            "virtual": VirtualLocalLibrary,
            "az": BlobExpertLibrary,
            "hf": HFExpertLibrary,
        }

        if expert_library_type in available_libraries:
            expert_lib_class = available_libraries[expert_library_type]
        elif expert_library_type in available_libraries.values():
            expert_lib_class = expert_library_type
        else:
            # if repo_id includes "local://", "virtual://", "az://", "hf://"
            prefix = repo_id.split("://")
            if prefix[0] in available_libraries:
                expert_library_type = prefix[0]
                repo_id = prefix[1]
            else:
                expert_library_type = "hf"
        try:
            expert_lib_class = available_libraries[expert_library_type]
        except KeyError:
            raise ValueError(f"Unknown expert library type {expert_library_type}.")
        return expert_lib_class

    @classmethod
    def get_expert_library(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        model_name: Optional[str] = None,
        selection: Optional[List[str]] = None,
        exclude_selection: Optional[List[str]] = None,
        create: bool = False,
        ignore_sliced: bool = False,
        expert_library_type: Union[Type["ExpertLibrary"], str] = None,
        destination_id: Optional[str] = None,
    ) -> "ExpertLibrary":
        """Instantiate an ExpertLibrary from one of the available expert library types:
            - "local": LocalExpertLibrary,
            - "virtual": VirtualLocalLibrary,
            - "az": BlobExpertLibrary,
            - "hf": HFExpertLibrary,

        The expert library type is selected based on the following order of priority:
            1. If expert_library_type is provided, and is one of ExpertLibrary subclasses, uses it.
            2. If repo_id includes one of the following substrings: "local://", "virtual://", "az://", "hf://".
            3. Otherwise, uses LocalExpertLibrary.

        If a destination_id is provided, the expert library will be copied to the new destination.
        """
        expert_lib_class = cls._get_expert_lib_class(repo_id, expert_library_type)
        expert_lib = expert_lib_class(
            repo_id=repo_id,
            token=token,
            model_name=model_name,
            selection=selection,
            exclude_selection=exclude_selection,
            create=create,
            ignore_sliced=ignore_sliced,
        )
        if destination_id is not None:
            expert_lib_class_copy = cls._get_expert_lib_class(
                destination_id, expert_library_type
            )
            expert_lib = expert_lib_class_copy.from_expert_library(
                expert_lib, destination_id, force=True, upload_aux_data=True
            )

        return expert_lib


class LocalExpertLibrary(ExpertLibrary, LocalFSEngine):
    """A local library stored on disk."""

    def add_expert(
        self, expert_dump: Expert, expert_name: str = None, force: bool = False
    ):
        expert_name = expert_name or expert_dump.expert_info.expert_name
        if "/" in expert_name:
            # create sub-folders if necessary
            path = expert_name.split("/")
            os.makedirs(os.path.join(self.repo_id, *path[:-1]), exist_ok=True)
        return super().add_expert(expert_dump, expert_name=expert_name, force=force)


class BlobExpertLibrary(ExpertLibrary, BlobStorageEngine):
    """Library stored in Azure Blob Storage."""

    pass


class HFExpertLibrary(ExpertLibrary, HuggingfaceHubEngine):
    """Library stored in Hugging Face Hub."""

    pass


class VirtualLocalLibrary(ExpertLibrary, LocalFSEngine):
    """
    A virtual library is not stored on disk, but only in memory.
    Useful for temporary library objects used during runtime.
    """

    def create_repo(self, repo_id, repo_type, exist_ok, private=True):
        pass

    def delete_repo(self, repo_id, repo_type=None):
        pass

    def _upload_metadata(self, metadata):
        pass

    def _upload_weights(self, expert_name, expert_dump):
        pass

    def _update_readme(self):
        pass


def get_best_expert_for_score(library: HFExpertLibrary, hash) -> Expert:
    best_expert = None
    best_score = -np.inf
    for metadata in library.data.values():
        score: Score = library.get_score(metadata.expert_name, hash=hash)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_expert = metadata
    return library[best_expert.expert_name] if best_expert is not None else None


def get_best_expert_for_task(library: HFExpertLibrary, task, hash) -> Expert:
    """
    Return the expert with the highest score on task. If none found, returns the last expert found.
    """
    if task not in library.tasks:
        raise ValueError(f"Task {task} not found in repository.")

    best_expert = None
    best_score = -np.inf
    for metadata in library.data.values():
        # if metadata.expert_task_name != task:
        #     continue
        score: Score = library.get_score(metadata.expert_name, hash=hash)
        if score is None:
            if metadata.expert_task_name == task:
                best_expert = metadata
            continue
        if score > best_score:
            best_score = score
            best_expert = metadata
    assert best_expert is not None
    return library[best_expert.expert_name]


def get_task_expert(task, expert_lib, default_score):
    """
    Get the best expert for a given task.

    Args:
        task (str): The task for which to find the expert.
        expert_lib (ExpertLibrary): The library of available experts.
        default_score (Score): Score to use for expert retrieval.

    Returns:
        Expert: The best expert for the given task according to the score.
    Raises:
        ValueError: If no default score is provided.
    """
    if default_score is None:
        raise ValueError("No default score provided")
    parent_exp: Expert = get_best_expert_for_score(expert_lib, default_score.hash)
    if parent_exp is None and task in expert_lib.tasks:
        parent_exp = get_best_expert_for_task(expert_lib, task, default_score.hash)
    return parent_exp
