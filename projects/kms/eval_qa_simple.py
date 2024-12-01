import copy
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning_fabric import seed_everything
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import (
    ExpertModel,
    ExpertModelConfig,
    MultiExpertModel,
    MultiExpertModelConfig,
    disable_modifiers,
)
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.utils import transfer_batch_to_device
from mttl.utils import create_library, remote_login, upload_library
from projects.kms.train_km_simple import (
    evaluate_class,
    evaluate_datasets,
    evaluate_metrics,
)

# register this datamodule!
from projects.kms.utils.km_datamodule import KMDatasetModule
from projects.kms.utils.nqa_datamodule import NQADatamodule  # noqa: F401
from projects.kms.utils.nqa_evaluator import NQAZeroShotEvaluator
from projects.kms.utils.simple_utils import SimpleLogger, dcd_loss, do_evaluation

torch.set_float32_matmul_precision("high")


@dataclass
class EvalArguments(ExpertConfig):
    km_path: str = None
    evaluate_on: str = "nqa"


def eval_qa(eval_args: EvalArguments):
    seed_everything(eval_args.seed, workers=True)

    # get directory of the current file
    setup_logging(eval_args.output_dir)

    logger.info("Args: %s", eval_args.to_json())

    remote_login(eval_args.remote_token)

    data_args = copy.deepcopy(eval_args)
    data_args.dataset = evaluate_datasets[eval_args.evaluate_on]
    evaluator = evaluate_class[eval_args.evaluate_on](data_args)

    if eval_args.km_path is not None:
        selection = evaluator.datamodule.test_dataset["document_id"]
        library = ExpertLibrary.get_expert_library(
            eval_args.km_path, selection=selection
        )
        model = MultiExpertModel.from_pretrained_library(
            library,
            device_map=get_device(),
            precision="bf16",
        )
    else:
        model = MultiExpertModel(
            MultiExpertModelConfig(base_model=eval_args.model),
            device_map=get_device(),
            precision="bf16",
        )

    rougeL = evaluator.evaluate(model, split="test")
    print(rougeL)


if __name__ == "__main__":
    args = EvalArguments.parse(raise_error=False)
    eval_qa(args)
