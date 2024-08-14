import os
from dataclasses import dataclass
from functools import partial

import numpy

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
    apply_source_template,
)
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.logging import logger
from mttl.models.library.expert_library import DatasetLibrary


@dataclass
class ClusterDataConfig(FlatMultiTaskConfig):
    """Just adapts the FlatMultiTaskConfig to a dataset containing a column with a cluster id."""

    task_name_field: str = "cluster_id"
    task_id_field: str = "cluster_id"


@DataModule.register("cluster_flat_multitask", ClusterDataConfig)
class ClusterDataModule(FlatMultiTaskModule):
    pass
