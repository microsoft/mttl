import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from collections import Counter

from mttl.utils import logger
from torch import nn

from sklearn.utils.extmath import safe_sparse_dot
from huggingface_hub import (
    create_repo,
    login,
    CommitOperationAdd,
    create_commit,
    preupload_lfs_files,
)

import torch


try:
    import faiss
except:
    logger.warn("Faiss not installed. KATE router will not work.")


def upload_checkpoint(repo_id, filename, path_in_repo):
    import os

    login(os.environ["HF_TOKEN"])
    create_repo(repo_id, repo_type="model", exist_ok=True)
    additions = [
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=filename),
    ]
    preupload_lfs_files(repo_id, additions=additions)
    return create_commit(
        repo_id, additions, commit_message=f"Add {filename} to repo {repo_id}."
    )


class TFIDFRouter:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.dataset_name = kwargs.get("dataset_name")
        self.vectorizer = None

    def train(self):
        import tqdm

        self.dataset = (
            load_dataset(self.dataset_name)["train"].shuffle().select(range(500_000))
        )
        self.vectorizer = TfidfVectorizer(
            norm="l2", sublinear_tf=True, stop_words="english"
        )
        self.train_features = self.vectorizer.fit_transform(
            tqdm.tqdm(self.dataset["source"])
        )
        self.task_names = list(self.dataset["task_name"])

    def predict_task(self, query):
        query = self.vectorizer.transform([query])
        results = safe_sparse_dot(query, self.train_features.T, dense_output=True)
        results = results.argsort()[0][::-1]
        results = {"task_name": [self.task_names[int(i)] for i in results[:100]]}
        return Counter(results["task_name"]).most_common(3)

    def state_dict(self):
        return {
            "config": self.config,
            "vectorizer": self.vectorizer,
            "train_features": self.train_features,
            "train_task_names": self.task_names,
        }

    def load_state_dict(self, state_dict):
        self.config = state_dict["config"]
        self.vectorizer = state_dict["vectorizer"]
        self.train_features = state_dict["train_features"]
        self.task_names = state_dict["train_task_names"]

    def save_pretrained(self, path, repo_id=None):
        import os

        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path + "/model.ckpt",
        )
        if repo_id:
            upload_checkpoint(repo_id, path + "/model.ckpt", "model.ckpt")

    @classmethod
    def from_pretrained(cls, repo_id):
        import os
        from huggingface_hub import hf_hub_download

        if not os.path.exists(repo_id):
            ckpt_file = hf_hub_download(repo_id, filename="model.ckpt")
        else:
            ckpt_file = repo_id + "/model.ckpt"

        ckpt = torch.load(ckpt_file)

        ranker = cls(**ckpt["config"])
        ranker.load_state_dict(ckpt["state_dict"])
        return ranker


class KATERouter:
    def __init__(self, **kwargs):
        from sentence_transformers import SentenceTransformer

        self.config = kwargs
        self.available_tasks = None
        self.dataset_name = kwargs.get("dataset_name")
        self.embedder = SentenceTransformer("all-mpnet-base-v2")

    def train(self):
        self.dataset = (
            load_dataset(self.dataset_name)["train"].shuffle().select(range(1_000_000))
        )
        self.train_features = self.embedder.encode(
            self.dataset["source"],
            show_progress_bar=True,
            batch_size=1024,
            device="cuda:0",
        )
        self.train_features = self.train_features / (
            (self.train_features**2).sum(axis=1, keepdims=True) ** 0.5
        )
        self.index = faiss.IndexFlatL2(self.train_features.shape[1])
        self.index.add(self.train_features)
        self.task_names = list(self.dataset["task_name"])

    def predict_batch(self, batch):
        query = self.embedder.encode(
            batch["sources_texts"], show_progress_bar=False, device="cuda:0"
        )
        _, indices = self.index.search(query, 200)
        top_tasks, top_weights = [], []
        for _, idxs in enumerate(indices):
            task_names = [self.task_names[int(idx)] for idx in idxs]
            top_selected = Counter(task_names).most_common(3)
            top_tasks.append(
                [
                    x[0]
                    for x in top_selected
                    if self.available_tasks is None or x[0] in self.available_tasks
                ]
            )
            ws = [
                x[1]
                for x in top_selected
                if self.available_tasks is None or x[0] in self.available_tasks
            ]
            top_weights.append(np.asarray(ws) / (np.asarray(ws).sum() + 1e-6))
        return top_tasks, top_weights

    def predict_task(self, query):
        query = self.embedder.encode([query], show_progress_bar=False, device="cuda:0")
        _, indices = self.index.search(query, 200)
        results = [self.task_names[int(i)] for i in indices[0]]
        top_selected = Counter(results).most_common(3)
        tasks = [
            x[0]
            for x in top_selected
            if self.available_tasks is None or x[0] in self.available_tasks
        ]
        weights = [
            x[1]
            for x in top_selected
            if self.available_tasks is None or x[0] in self.available_tasks
        ]
        return tasks, weights

    def state_dict(self):
        return {
            "config": self.config,
            "train_task_names": self.task_names,
        }

    def load_state_dict(self, state_dict):
        self.config = state_dict["config"]
        self.task_names = state_dict["train_task_names"]

    def set_available_tasks(self, available_tasks):
        self.available_tasks = available_tasks

    def save_pretrained(self, path, repo_id=None):
        import os
        from faiss import write_index

        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path + "/model.ckpt",
        )
        write_index(self.index, path + "/index.faiss")

        if repo_id:
            upload_checkpoint(repo_id, path + "/model.ckpt", "model.ckpt")
            upload_checkpoint(repo_id, path + "/index.faiss", "index.faiss")

    @classmethod
    def from_pretrained(cls, path):
        import os
        from huggingface_hub import hf_hub_download
        from faiss import read_index

        if not os.path.exists(path):
            ckpt_file = hf_hub_download(path, filename="model.ckpt")
            index_file = hf_hub_download(path, filename="index.faiss")
        else:
            ckpt_file = path + "/model.ckpt"
            index_file = path + "/index.faiss"

        ckpt = torch.load(ckpt_file)
        index = read_index(index_file)

        ranker = cls(**ckpt["config"])
        ranker.load_state_dict(ckpt["state_dict"])
        ranker.index = index
        return ranker
