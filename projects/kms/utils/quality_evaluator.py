from typing import Dict

from mttl.arguments import create_config_class_from_args
from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.logging import warn_once
from projects.kms.utils.nqa_datamodule import NQADatamodule, NQADatasetConfig
from projects.kms.utils.quality_datamodule import (
    QualityDatamodule,
    QualityDatasetConfig,
)


class QualityEvaluator(LogLikeEvaluator):
    def __init__(self, dataset_args: "DataArgs", generation_kwargs: Dict = {}):
        from mttl.datamodule.base import get_datamodule

        dataset_args.dataset_type = "quality"
        dataset_args.add_eos_to_targets = False

        datamodule = get_datamodule(dataset_args, for_generation=False)
        super().__init__(
            datamodule, generation_kwargs=generation_kwargs, length_normalization=True
        )

    def evaluate(self, model, split="dev", **kwargs):
        return super().evaluate(model, split=split, **kwargs)
