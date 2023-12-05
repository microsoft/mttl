from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from collections import Counter

from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
from sklearn.utils.extmath import safe_sparse_dot
import torch
import faiss


class Router:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.dataset = load_dataset(kwargs.get("dataset_name"))
        self.vectorizer = None

    def train(self):
        raise NotImplementedError

    def predict_task(self, query):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def save_pretrained(self, path, repo_id=None):
        raise NotImplementedError

    def from_pretrained(self, path):
        raise NotImplementedError


class TFIDFRouter(Router):
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
        from huggingface_hub import upload_file

        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path + "/model.ckpt",
        )
        if repo_id:
            upload_file(path, repo_id)

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


class KATERouter(Router):
    def __init__(self, **kwargs):
        from sentence_transformers import SentenceTransformer

        self.config = kwargs
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

    def predict_task(self, query):
        query = self.embedder.encode([query], show_progress_bar=False, device="cuda:0")
        d, indices = self.index.search(query, 100)
        results = {"task_name": [self.task_names[int(i)] for i in indices[0][:100]]}
        return Counter(results["task_name"]).most_common(3)

    def state_dict(self):
        return {
            "config": self.config,
            "train_task_names": self.task_names,
        }

    def load_state_dict(self, state_dict):
        self.config = state_dict["config"]
        self.task_names = state_dict["train_task_names"]

    def save_pretrained(self, path, repo_id=None):
        import os
        from huggingface_hub import upload_file
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
            import huggingface_hub

            huggingface_hub.login(os.environ["HF_TOKEN"])
            huggingface_hub.create_repo(repo_id, repo_type="model", exist_ok=True)
            additions = [
                huggingface_hub.CommitOperationAdd(
                    "model.ckpt", path=path + "/model.ckpt"
                ),
                huggingface_hub.CommitOperationAdd(
                    "index.faiss", path=path + "/index.faiss"
                ),
            ]
            huggingface_hub.preupload_lfs_files(repo_id, additions=additions)
            huggingface_hub.create_commit(
                repo_id, additions, commit_message="Add model.ckpt and index.faiss."
            )

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


def retrieve_mmlu(dataset_name="sordonia/adauni-v3-10k-flat", force=True):
    import os

    if os.path.exists("/data/alsordon/kate-router") and not force:
        router = KATERouter.from_pretrained("/data/alsordon/kate-router")
    else:
        router = KATERouter(dataset_name=dataset_name)
        router.train()
        router.save_pretrained("/data/alsordon/kate-router")

    datamodule = MMLUDataModule(
        MMLUDataConfig(
            "mmlu",
            model="phi-2",
            max_input_length=2048,
            max_output_length=128,
            predict_batch_size=1,
        )
    )

    test_loader = datamodule.test_dataloader(shuffle=True)
    for batch in test_loader:
        text = batch["sources_texts"][0]
        print(batch["task_names"][0])
        results = router.predict_task(text)
        print("Best tasks:", results)


retrieve_mmlu()
