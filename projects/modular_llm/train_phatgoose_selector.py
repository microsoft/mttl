import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.library.expert_library import ExpertLibrary
from pytorch_lightning import seed_everything
from mttl.models.expert_config import ExpertConfig


def parse_libname(libname):
    parts = libname.split("|")
    if len(parts) == 1:
        return libname, None
    else:
        return parts[0], parts[-1].split(",")


def train_with_transform(args: ExpertConfig):
    seed_everything(args.seed, workers=True)
    from mttl.models.library.library_transforms import (
        PhatgooseTransform,
        PhatgooseConfig,
    )

    library_id, expert_names = parse_libname(args.library_id)
    library = ExpertLibrary.get_expert_library(
        repo_id=library_id,
        create=False,
        destination_id=args.destination_library_id,
    )
    phagoose_transform = PhatgooseTransform(
        PhatgooseConfig(n_steps=args.n_steps_pg, learning_rate=args.learning_rate_pg)
    )
    embeddings = phagoose_transform.transform(
        library, expert_names=expert_names, default_args=args, recompute=True
    )
    print(len(embeddings))


if __name__ == "__main__":
    args = ExpertConfig.parse()
    train_with_transform(args)
