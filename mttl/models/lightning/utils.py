import hashlib
import json
import os
import re
from collections import defaultdict, deque
from enum import Enum
from typing import Callable, Optional, Union

import prettytable
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from transformers.file_utils import PushToHubMixin
from transformers.utils import cached_file

from mttl.logging import logger
from mttl.models.get_optimizer import get_optimizer
from mttl.models.get_scheduler import get_scheduler
from mttl.utils import get_checkpoint_path


def convert_hps_to_dict(hparams):
    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        v = v.name if isinstance(v, Enum) else v
        hparams_allowed[k] = v
    return hparams_allowed
