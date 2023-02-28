import pickle
import numpy as np
import torch


class ClusterResult:
    _instance = None

    @classmethod
    def get_distances(cls, hash):
        return cls._instance._example_to_distances[hash]

    @classmethod
    def get_distances_batch(cls, hashes):
        return [cls.get_distances(h) for h in hashes]

    @classmethod
    def get_cluster(cls, hash):
        return cls._instance.example_to_ids[hash]

    @classmethod
    def n_clusters(cls):
        if "cluster_distances" in cls._instance._raw_data:
            return len(cls._instance._raw_data["cluster_distances"][0])
        else:
            return max(cls._instance.example_to_ids.values())

    def __new__(cls, path):
        if cls._instance is None:
            cls._instance = super(ClusterResult, cls).__new__(cls)

            with open(path, "rb") as f:
                cls._instance._raw_data = pickle.load(f)

            if "hashes" not in cls._instance._raw_data:
                # backward compatibility
                cls._instance.example_to_ids = cls._instance._raw_data
            else:
                cls._instance.example_to_ids = dict(
                    zip(
                        cls._instance._raw_data["hashes"],
                        cls._instance._raw_data["cluster_ids"],
                    )
                )
                cls._instance._example_to_distances = dict(
                    zip(
                        cls._instance._raw_data["hashes"],
                        list(cls._instance._raw_data["cluster_distances"]),
                    )
                )
                cluster_sizes = torch.from_numpy(np.bincount(cls._instance._raw_data["cluster_ids"])).float()
                cls._instance.cluster_sizes = cluster_sizes
                cls._instance.avg_cluster_size = cluster_sizes.mean().item()
        return cls._instance
