import os
import json
from mttl.models.modifiers.expert_containers.expert_library import (
    HFExpertLibrary,
    LocalExpertLibrary,
)
from mttl.models.modifiers.expert_containers.library_transforms import (
    MBClusteringTransformConfig,
    MBCWithCosSimTransform,
)
from mttl.utils import logger
from mttl.models.expert_config import ExpertConfig


def main(args: ExpertConfig):
    library = HFExpertLibrary(args.library_id)

    # making local copy of remote lib
    destination = os.environ.get(
        "HF_LIB_CACHE", os.path.expanduser("~/.cache/huggingface/libraries")
    )
    destination += args.library_id
    os.makedirs(destination, exist_ok=True)
    library = LocalExpertLibrary.from_expert_library(library, repo_id=destination)

    cfg = MBClusteringTransformConfig(
        k=args.mbc_num_clusters,
        random_state=42,
        sparsity_threshold=0.1,
        recompute_embeddings=False,
    )
    transform = MBCWithCosSimTransform(cfg)
    clusters = transform.transform(library)

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
