import os
import json
from mttl.models.modifiers.expert_containers.expert_library import ExpertLibrary
from mttl.models.modifiers.expert_containers.library_transforms import (
    MBClusteringTransformConfig,
    MBCWithCosSimTransform,
)
from mttl.utils import logger
from mttl.models.expert_config import ExpertConfig


def main(args: ExpertConfig):
    library = ExpertLibrary.get_expert_library(
        repo_id=args.library_id,
        create=False,
        destination_id=args.destination_library_id,
    )

    cfg = MBClusteringTransformConfig(
        k=args.mbc_num_clusters, random_state=42, sparsity_threshold=0.5
    )
    transform = MBCWithCosSimTransform(cfg)
    clusters = transform.transform(library, recompute=True)

    output_json_file = (
        f"{os.path.dirname(os.path.realpath(__file__))}/task_sets/{args.library_id}/"
    )
    os.makedirs(output_json_file, exist_ok=True)
    filename = f"{args.mbc_num_clusters}MBC.json"
    cluster_dict = {}
    for c, l in clusters.items():
        print(f"Cluster {c} has {len(l)} elements")
        print(f"c{c}o{args.mbc_num_clusters} = {l}")
        cluster_dict[f"c{c}o{args.mbc_num_clusters}"] = l
    with open(output_json_file + f"/{filename}", "w") as f:
        json.dump(cluster_dict, f, indent=4)
    logger.info(f"Saved clusters to {output_json_file}/{filename}")


if __name__ == "__main__":
    args = ExpertConfig.parse()
    main(args)
