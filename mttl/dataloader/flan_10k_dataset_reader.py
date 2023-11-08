import torch
import datasets
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


class Flan10kDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, category, idxs=None):
        super().__init__()
        self.dataset = datasets.load_dataset("sordonia/flan-10k")[category]
        self.task_name = category
        if idxs is not None:
            self.dataset = self.dataset.select(idxs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        entry = self.dataset[key]

        source = entry["source"]
        target = entry["target"]
        task_id = -1
        return {
            "source": source,
            "target": target,
            "task_id": task_id,
            "task_name": self.task_name,
        }


if __name__ == "__main__":
    dataset = Flan10kDataset()
    print(dataset[0])
