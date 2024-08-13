import os
import sys

import torch
from pytorch_lightning import Trainer, seed_everything

from mttl.callbacks import (
    DownstreamEvalCallback,
    LiveCheckpointCallback,
    NanoMMLUCallback,
    RougeCallback,
)
from mttl.cli.train_experts import train_experts
from mttl.config import MoEExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.logging import get_pl_loggers, logger, setup_logging
from mttl.models.expert_model import MoEModel
from mttl.models.monitors import get_monitors
from mttl.utils import remote_login

if __name__ == "__main__":
    train_experts(MoEExpertConfig.parse(), MoEModel)
