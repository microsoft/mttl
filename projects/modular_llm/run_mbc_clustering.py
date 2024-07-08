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
        self.output_file = None


def main(args: ClusteringConfig):
    if args.output_file is None:
        raise ValueError("Please provide an output file.")

    library = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        create=False,
    )

    if args.cluster_mode == "mbc":
        cfg = MBClusteringTransformConfig(
            k=args.num_clusters, random_state=42, sparsity_threshold=0.5
        )
        transform = MBCWithCosSimTransform(cfg)
        clusters = transform.transform(library, recompute=True)
    elif args.cluster_mode == "random":
        cfg = RandomClustersConfig(k=args.num_clusters, random_state=42)
        transform = RandomClustersTransform(cfg)
        clusters = transform.transform(library)
    else:
        raise ValueError(f"Unknown cluster mode {args.cluster_mode}")

    os.makedirs(os.path.basedir(args.output_file), exist_ok=True)
    cluster_dict = {}
    for c, l in clusters.items():
        print(f"Cluster {c} has {len(l)} elements")
        print(f"c{c}o{args.num_clusters} = {l}")
        cluster_dict[f"c{c}o{args.num_clusters}"] = l
    with open(args.output_file, "w") as f:
        json.dump(cluster_dict, f, indent=4)
    logger.info(f"Saved clusters to {args.output_file}")


if __name__ == "__main__":
    args = ClusteringConfig.parse()
    main(args)
