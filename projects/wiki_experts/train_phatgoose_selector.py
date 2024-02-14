import os
import sys
import pytorch_lightning as pl
import glob
import copy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.models.modifiers.expert_containers.expert_library import get_expert_library
from mttl.callbacks import LiveCheckpointCallback

from mttl.models.monitors import get_monitors
from projects.wiki_experts.src.callbacks import DownstreamEvalCallback


import torch
from pytorch_lightning import Trainer, seed_everything

from projects.wiki_experts.utils import get_datamodule
from mttl.callbacks import NanoMMLUCallback, RougeCallback
from mttl.utils import (
    get_pl_loggers,
    remote_login,
    setup_logging,
    logger,
)
from huggingface_hub import whoami
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
from projects.wiki_experts.src.expert_model import MultiExpertModel
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.evolution.transfer_matrix import (
    TransferMatrixConfig,
    run_eval as produce_transfer_matrix,
)


def parse_libname(libname):
    parts = libname.split("/")
    if len(parts) == 2:
        return libname, None
    else:
        return "/".join(parts[:-1]), parts[-1].split(",")


def train_with_transform(args: ExpertConfig):
    seed_everything(args.seed, workers=True)
    from mttl.models.modifiers.expert_containers.library_transforms import (
        PhatgooseTransform,
        PhatgooseConfig,
    )

    library_id, expert_names = parse_libname(args.library_id)
    library = get_expert_library(library_id, create=False)
    phagoose_transform = PhatgooseTransform(PhatgooseConfig(recompute=False))
    embeddings = phagoose_transform.transform(
        library, expert_names=expert_names, default_args=args
    )
    print(len(embeddings))


if __name__ == "__main__":
    args = ExpertConfig.parse()
    train_with_transform(args)
