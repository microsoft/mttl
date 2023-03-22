import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Encodings:
    encodings: List = field(default_factory=list)
    hashes: List = field(default_factory=list)
    task_names: List = field(default_factory=list)
    task_ids: List = field(default_factory=list)
    is_test: List = field(default_factory=list)
    input_type: str = "input"

    def clear(self):
        self.encodings.clear()
        self.hashes.clear()
        self.task_names.clear()
        self.task_ids.clear()
        self.is_test.clear()

    @staticmethod
    def load(save_path):
        import pickle

        with open(save_path, "rb") as f:
            data = pickle.load(f)

        return Encodings(
            data["encodings"].tolist(),
            data["hashes"],
            data["task_names"],
            data["task_ids"],
            data["is_test"],
            data["input_type"],
        )

    def save(self, save_path):
        import pickle

        with open(save_path, "wb") as f:
            encodings = np.array(self.encodings).astype(np.float32)
            print("Saving chunk: ", save_path, " with length: ", len(self.encodings))

            pickle.dump(
                {
                    "encodings": encodings,
                    "hashes": self.hashes,
                    "task_names": self.task_names,
                    "task_ids": self.task_ids,
                    "is_test": self.is_test,
                    "input_type": self.input_type,
                },
                f,
            )


@dataclass
class ClusterInfos:
    hashes: List = field(default_factory=list)
    cluster_ids: List = field(default_factory=list)
    centroids: List = field(default_factory=list)
    cluster_dists: List = field(default_factory=list)
    task_names: List = field(default_factory=list)
    is_test: List = field(default_factory=list)
    input_type: str = "input"

    @staticmethod
    def load(save_path):
        """Load all fields from a pickle file
        """
        import pickle

        with open(save_path, "rb") as f:
            data = pickle.load(f)

        return ClusterInfos(
            hashes=data["hashes"],
            cluster_ids=data["cluster_ids"],
            cluster_dists=data["cluster_distances"],
            centroids=data["centroids"],
            task_names=data["task_names"],
            is_test=data["is_test"],
            input_type=data["input_type"]
        )   

    def save(self, save_path):
        import pickle

        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "hashes": self.hashes,
                    "cluster_ids": self.cluster_ids,
                    "cluster_distances": self.cluster_dists,
                    "centroids": self.centroids,
                    "task_names": self.task_names,
                    "is_test": self.is_test,
                    "input_type": self.input_type
                },
                f,
            )
