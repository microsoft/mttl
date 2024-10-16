from nqa_datamodule import NQADatamodule, NQADatasetConfig

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.evaluators.rouge_evaluator import RougeEvaluator


class NQAZeroShotEvaluator(RougeEvaluator):
    def __init__(self, dataset_args, generation_kwargs):
        datamodule = NQADatamodule(
            NQADatasetConfig(
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

    def evaluate(self, model, split="test", **kwargs):
        if self.datamodule.test_dataset and len(self.datamodule.test_dataset):
            split_ = "train"
        elif self.datamodule.dev_dataset and len(self.datatamodule.dev_dataset):
            split_ = "dev"
        else:
            split_ = "test"

        # when selecting on document ids, some dataset will be empty,
        # to make the evaluators not complain, we just set the dataset to the first one that is not empty
        dataset_ = getattr(self.datamodule, f"{split}_dataset")
        if not dataset_:
            split = split_

        return super().evaluate(model, split=split, **kwargs)
