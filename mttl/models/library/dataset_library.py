import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from mttl.models.library.backend_engine import (
    BackendEngine,
    BlobStorageEngine,
    HuggingfaceHubEngine,
    LocalFSEngine,
)
from mttl.utils import retry


class DatasetEngine(ABC):
    def __init__(self, dataset_id: str, token: Optional[str] = None):
        self.dataset_id = dataset_id
        self.token = token

    @property
    def backend_engine(self) -> BackendEngine:
        backend_engine: BackendEngine = self._backend_engine()
        backend_engine.login(self.token)
        return backend_engine

    @property
    @abstractmethod
    def _backend_engine(self) -> Type[BackendEngine]:
        pass

    @abstractmethod
    def pull_dataset(
        self,
        name: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dataset:
        pass

    @abstractmethod
    def push_dataset(
        self,
        dataset: Union[DatasetDict, Dataset],
        name: Optional[str] = None,
    ) -> None:
        pass

    def delete_dataset(self) -> None:
        self.backend_engine.delete_repo(self.dataset_id, repo_type="dataset")

    def _concat_paths(self, *args) -> str:
        """Concatenate paths ordered as received. Ignore None values."""
        results_path = Path()
        for p in args:
            if p is not None:
                results_path /= p
        return str(results_path)


class HuggingfaceHubDatasetEngine(DatasetEngine):
    @property
    def _backend_engine(self) -> Type[BackendEngine]:
        return HuggingfaceHubEngine

    def pull_dataset(
        self,
        name: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs,
    ) -> Dataset:
        return load_dataset(
            self.dataset_id, name, split=split, trust_remote_code=True, **kwargs
        )

    def push_dataset(
        self,
        dataset: Union[DatasetDict, Dataset],
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        dataset.push_to_hub(self.dataset_id, name, **kwargs)


class LocalDatasetEngine(DatasetEngine):
    @property
    def _backend_engine(self) -> Type[BackendEngine]:
        return LocalFSEngine

    def pull_dataset(
        self,
        name: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dataset:
        # Saves the dataset subset and split as subdirectories if provided
        dataset_path = self._concat_paths(self.dataset_id, name, split)
        dataset = load_from_disk(dataset_path)
        return dataset

    def push_dataset(
        self,
        dataset: Union[DatasetDict, Dataset],
        name: Optional[str] = None,
    ) -> None:
        dataset_path = self._concat_paths(self.dataset_id, name)
        self.backend_engine.create_repo(
            repo_id=self.dataset_id, repo_type="dataset", exist_ok=True
        )
        # HF push_to_hub sets the split to "train" if it's None
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        dataset.save_to_disk(dataset_path)


class BlobStorageDatasetEngine(DatasetEngine):
    @property
    def _backend_engine(self) -> Type[BackendEngine]:
        return BlobStorageEngine

    def pull_dataset(
        self,
        name: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dataset:
        local_path = self._concat_paths(self.dataset_id, name, split)
        download_filter = self._concat_paths(local_path, "*")
        self.backend_engine.snapshot_download(self.dataset_id, download_filter)
        dataset_cache_dir = str(
            self.backend_engine.get_repository_cache_dir(self.dataset_id)
        )
        dataset_cache_dir = self._concat_paths(dataset_cache_dir, name, split)
        dataset = load_from_disk(dataset_cache_dir)
        return dataset

    def push_dataset(
        self,
        dataset: Union[DatasetDict, Dataset],
        name: Optional[str] = None,
    ) -> None:
        self.backend_engine.create_repo(
            repo_id=self.dataset_id, repo_type="dataset", exist_ok=True
        )
        # HF push_to_hub sets the split to "train" if it's None
        if isinstance(dataset, Dataset):
            dataset = DatasetDict({"train": dataset})
        dataset_cache_dir = str(
            self.backend_engine.get_repository_cache_dir(self.dataset_id)
        )
        # Name is a subset of the dataset. Save in its own directory
        dataset_path = self._concat_paths(dataset_cache_dir, name)
        dataset.save_to_disk(dataset_path)

        asyncio.run(
            self.backend_engine.async_upload_folder(
                self.dataset_id, dataset_cache_dir, recursive=True
            )
        )


class DatasetLibrary:
    @classmethod
    def pull_dataset(
        cls, dataset_id: str, token: Optional[str] = None, **kwargs
    ) -> Dataset:
        dataset_engine = cls._get_dataset_engine(dataset_id, token)
        dataset = dataset_engine.pull_dataset(**kwargs)
        return dataset

    @classmethod
    @retry(max_retries=5, wait_seconds=60)
    def pull_dataset_with_retry(
        cls, dataset_id: str, token: Optional[str] = None, **kwargs
    ) -> Union[DatasetDict, Dataset]:
        return cls.pull_dataset(dataset_id, token, **kwargs)

    @classmethod
    def push_dataset(
        cls,
        dataset: Union[DatasetDict, Dataset],
        dataset_id: str,
        token: Optional[str] = None,
        **kwargs,
    ) -> None:
        dataset_engine = cls._get_dataset_engine(dataset_id, token)
        dataset_engine.push_dataset(dataset, **kwargs)

    @classmethod
    def delete_dataset(cls, dataset_id: str, token: Optional[str] = None) -> None:
        dataset_engine = cls._get_dataset_engine(dataset_id, token)
        dataset_engine.delete_dataset()

    @staticmethod
    def _get_dataset_engine(dataset_id: str, token: Optional[str]) -> DatasetEngine:
        engines = {
            "local": LocalDatasetEngine,
            "az": BlobStorageDatasetEngine,
            "hf": HuggingfaceHubDatasetEngine,
        }
        prefix = dataset_id.split("://")
        if prefix[0] in engines:
            engine_id = prefix[0]
            dataset_id = prefix[1]
        else:
            engine_id = "hf"
        try:
            engine = engines[engine_id](dataset_id=dataset_id, token=token)
            return engine
        except KeyError:
            raise ValueError(
                f"Unknown dataset engine type {engine_id}. "
                f"Available engines: {list(engines.keys())}"
            )
