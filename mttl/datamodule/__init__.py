from collections import defaultdict
from torch.utils.data.dataset import ConcatDataset
import numpy as np


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
            tasks_to_ids[task], max(len(tasks_to_ids[task]) // n, 1), replace=False
        ).tolist()
    return indices


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
        example_info["example_id"] = idx
        return example_info
