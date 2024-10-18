from nqa_datamodule import (
    MiniNQADatamodule,
    MiniNQADatasetConfig,
    NQADatamodule,
    NQADatasetConfig,
)

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.evaluators.rouge_evaluator import RougeEvaluator


class NQAZeroShotEvaluator(RougeEvaluator):
    def __init__(self, dataset_args, generation_kwargs):
        if dataset_args.dataset_type == "mini_nqa":
            dm_cls, dm_cfg_cls = MiniNQADatamodule, MiniNQADatasetConfig
        else:
            dm_cls, dm_cfg_cls = NQADatamodule, NQADatasetConfig

        datamodule = dm_cls(
            dm_cfg_cls(
                dataset=dataset_args.dataset,
                model=dataset_args.model,
                model_family=dataset_args.model_family,
                predict_batch_size=dataset_args.predict_batch_size,
                finetune_task_name=dataset_args.finetune_task_name,
                include_context=dataset_args.include_context,
                subsample_train=dataset_args.subsample_train,
                subsample_dev=dataset_args.subsample_dev,
                subsample_test=dataset_args.subsample_test,
            ),
            for_generation=True,
        )
        super().__init__(datamodule, generation_kwargs=generation_kwargs)

    def evaluate(self, model, split=None, **kwargs):

        # splits in order of preference
        splits = ["test", "dev", "train"]

        if split is not None:
            assert split in splits, f"Split {split} not found in {splits}"
            dataset = getattr(self.datamodule, f"{split}_dataset")
            assert dataset and len(
                dataset
            ), f"invalid dataset for split {split}, \n{dataset}"
        else:
            for split in splits:
                dataset = getattr(self.datamodule, f"{split}_dataset")
                if dataset and len(dataset):
                    break

        return super().evaluate(model, split=split, **kwargs)
