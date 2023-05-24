import numpy as np
import torch

from mttl.cluster_tuning.encodings import ClusterInfos


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
        return len(cls._instance.cluster_sizes)

    @classmethod
    def remove_skills(cls, skill_ids_to_keep):
        # print(skill_ids_to_keep)
        for key,v in cls._instance._example_to_distances.items():
            cls._instance._example_to_distances[key] = np.array(v)[skill_ids_to_keep].tolist()
    
    def __new__(cls, path):
        if cls._instance is None:
            cls._instance = super(ClusterResult, cls).__new__(cls)

            cluster_infos = ClusterInfos.load(path)

            cls._instance.infos = cluster_infos
            cls._instance.example_to_ids = dict(
                    zip(
                        cls._instance.infos.hashes,
                        cls._instance.infos.cluster_ids,
                    )
                )
            cls._instance._example_to_distances = dict(
                zip(
                    cls._instance.infos.hashes,
                    list(cls._instance.infos.cluster_dists),
                )
            )
            cluster_sizes = torch.from_numpy(np.bincount(cls._instance.infos.cluster_ids)).float()
            assert len(np.bincount(cls._instance.infos.cluster_ids)) == len(cls._instance._example_to_distances[list(cls._instance._example_to_distances.keys())[0]])

            cls._instance.cluster_sizes = cluster_sizes
            cls._instance.avg_cluster_size = cluster_sizes.mean().item()
        return cls._instance
