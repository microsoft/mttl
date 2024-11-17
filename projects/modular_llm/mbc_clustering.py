import json
import os

from mttl.arguments import Args
from mttl.logging import logger
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import (
    MBClusteringTransformConfig,
    MBCWithCosSimTransform,
    RandomClustersConfig,
    RandomClustersTransform,
)


class ClusteringConfig(Args):
    cluster_mode: str = "mbc"  # clustering mode: mbc, random
    output_file: str = None
    num_clusters: int = 10


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
