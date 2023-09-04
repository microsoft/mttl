from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import numpy as np
from pytorch_lightning import LightningDataModule
from abc import ABC, abstractmethod


def take_n_examples_per_task(task_names, n, rng=None):
    """Returns indices of x / n examples per task given a list of task names.

    Args:
        task_names (list): List of task names, one per example.
        n (int): Subsampling factor.
        rng (np.random.RandomState, optional): Random number generator. Defaults to None.

    Returns:
        list: List of indices.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    tasks_to_ids = defaultdict(list)
    for i, task in enumerate(task_names):
        tasks_to_ids[task].append(i)
    indices = []
    for task in tasks_to_ids.keys():
        indices += rng.choice(
            tasks_to_ids[task], min(len(tasks_to_ids[task]) // n, 1), replace=False
        ).tolist()
    return indices


class UniformSampler:
    def __init__(
        self, dataset: ConcatDataset, task_size: int, batch_size: int, **kwargs
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )

        self.dataset = dataset
        self.num_samples = len(dataset)
        self.num_tasks = len(dataset.datasets)
        self.task_size = task_size
        self.batch_size = batch_size
        self.nex_per_task = self.batch_size // self.task_size

        self.tasks_to_idx = []
        self.sizes = []
        total_idx = 0

        for _, dataset in enumerate(dataset.datasets):
            self.sizes.append(len(dataset))
            self.tasks_to_idx.append([i + total_idx for i in range(self.sizes[-1])])
            total_idx += self.sizes[-1]

        self.weights = np.array(self.sizes) / sum(self.sizes)
        self.batch_size = batch_size
        self.drop_last = True

    def __iter__(self):
        batch = []

        active_tasks = range(len(self.tasks_to_idx))
        indices_per_task = [
            np.random.permutation(len(self.tasks_to_idx[task])) for task in active_tasks
        ]
        for _ in range(len(self)):
            tasks = np.random.choice(
                active_tasks, self.task_size, p=self.weights
            ).tolist()
            for task in tasks:
                exs_idx = indices_per_task[task][: self.nex_per_task]
                indices_per_task[task] = indices_per_task[task][self.nex_per_task :]
                if len(indices_per_task[task]) < self.nex_per_task:
                    self.weights[active_tasks] = 0.0
                    self.weights /= self.weights.sum()
                batch.extend([self.tasks_to_idx[task][x] for x in exs_idx])
            yield batch

    def __len__(self) -> int:
        size = 0
        for s in self.sizes:
            size += self.nex_per_task * (s // self.nex_per_task)
        return size // self.batch_size


class TaskSampler:
    def __init__(
        self, dataset: ConcatDataset, task_size: int, batch_size: int, **kwargs
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )

        self.dataset = dataset
        self.num_samples = len(dataset)
        self.num_tasks = len(dataset.datasets)
        self.task_size = task_size
        self.batch_size = batch_size
        self.nex_per_task = self.batch_size // self.task_size

        self.tasks_to_idx = []
        self.sizes = []
        total_idx = 0

        for _, dataset in enumerate(dataset.datasets):
            self.sizes.append(len(dataset))
            self.tasks_to_idx.append([i + total_idx for i in range(self.sizes[-1])])
            total_idx += self.sizes[-1]

        print(self.sizes)
        self.weights = np.array(self.sizes) / sum(self.sizes)
        self.batch_size = batch_size
        self.drop_last = True

    def __iter__(self):
        batch = []

        active_tasks = range(len(self.tasks_to_idx))
        indices_per_task = [
            np.random.permutation(len(self.tasks_to_idx[task])) for task in active_tasks
        ]
        for _ in range(len(self)):
            if len(active_tasks) < self.task_size:
                break

            tasks = np.random.choice(
                active_tasks, self.task_size, p=self.weights
            ).tolist()
            for task in tasks:
                exs_idx = indices_per_task[task][: self.nex_per_task]
                indices_per_task[task] = indices_per_task[task][self.nex_per_task :]
                if len(indices_per_task[task]) < self.nex_per_task:
                    self.weights[active_tasks] = 0.0
                    self.weights /= self.weights.sum()
                batch.extend([self.tasks_to_idx[task][x] for x in exs_idx])
            yield batch

    def __len__(self) -> int:
        size = 0
        for s in self.sizes:
            size += self.nex_per_task * (s // self.nex_per_task)
        return size // self.batch_size


class TrainIndices:
    _instance = None

    def __init__(self, dataset, num_examples, seed):
        self.num_examples = num_examples
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.train_indices = self.rng.randint(
            0, len(dataset), self.num_examples
        ).tolist()

    @classmethod
    def get(cls, dataset, num_examples, seed):
        if cls._instance is None:
            cls._instance = cls(dataset, num_examples, seed)

        return cls._instance.train_indices


class IndexConcatDataset(ConcatDataset):
    def __getitem__(self, idx):
        example_info = super().__getitem__(idx)
        example_info.example_id = idx
        return example_info
