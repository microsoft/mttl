import os
import json
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.expert_containers.library_transforms import (
    MBClusteringTransformConfig,
    MBCWithCosSimTransform,
    RandomClustersTransform,
    RandomClustersConfig,
)
from mttl.utils import logger
from mttl.models.expert_config import ExpertConfig


class ClusteringConfig(ExpertConfig):
    def _set_defaults(self):
        super()._set_defaults()
        # for MBC
        self.num_clusters = 10  # number of clusters
        self.cluster_mode = "mbc"  # clustering mode: mbc, random


def main(args: ClusteringConfig):
    library = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        create=False,
        destination_id=args.destination_library_id,
    )

    if args.cluster_mode == "mbc":
        cfg = MBClusteringTransformConfig(
            k=args.num_clusters, random_state=42, sparsity_threshold=0.5
        )
        transform = MBCWithCosSimTransform(cfg)
        clusters = transform.transform(library, recompute=True)
        filename = f"{args.num_clusters}_mbc.json"
    elif args.cluster_mode == "random":
        cfg = RandomClustersConfig(k=args.num_clusters, random_state=42)
        transform = RandomClustersTransform(cfg)
        clusters = transform.transform(library)
        filename = f"{args.num_clusters}_random.json"
    else:
        raise ValueError(f"Unknown cluster mode {args.cluster_mode}")

    output_json_file = (
        f"{os.path.dirname(os.path.realpath(__file__))}/task_sets/{args.library_id}/"
    )
    os.makedirs(output_json_file, exist_ok=True)
    cluster_dict = {}
    for c, l in clusters.items():
        print(f"Cluster {c} has {len(l)} elements")
        print(f"c{c}o{args.num_clusters} = {l}")
        cluster_dict[f"c{c}o{args.num_clusters}"] = l
    with open(output_json_file + f"/{filename}", "w") as f:
        json.dump(cluster_dict, f, indent=4)
    logger.info(f"Saved clusters to {output_json_file}/{filename}")


if __name__ == "__main__":
    args = ClusteringConfig.parse()
    main(args)
