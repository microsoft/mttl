from nqa_datamodule import NQADatamodule, NQADatasetConfig

from mttl.arguments import create_config_class_from_args
from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.evaluators.rouge_evaluator import RougeEvaluator
from mttl.logging import warn_once


class NQAZeroShotEvaluator(RougeEvaluator):
    def __init__(self, dataset_args, generation_kwargs):

        from mttl.datamodule.base import get_datamodule

        # set the dataset type
        dataset_args.dataset_type = "narrativeqa"

        datamodule = get_datamodule(dataset_args, for_generation=True)

        super().__init__(datamodule, generation_kwargs=generation_kwargs)

    def evaluate(self, model, split=None, **kwargs):

        # splits in order of preference
        splits = ["test", "dev", "train"]

        if split is not None:
            assert split in splits, f"Split {split} not found in {splits}"
            dataset = getattr(self.datamodule, f"{split}_dataset")
            if dataset is None or len(dataset) == 0:
                warn_once(
                    f"invalid dataset for split {split}, \n{dataset}. Using default split."
                )
                split = None

        if split is None:
            for split in splits:
                dataset = getattr(self.datamodule, f"{split}_dataset")
                if dataset and len(dataset):
                    break

        return super().evaluate(model, split=split, **kwargs)
