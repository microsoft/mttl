from mttl.datamodule.hellaswag_data_module import HellaswagMultiChoiceDataModule
from mttl.evaluators.base import switch_to_eval_mode
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class HellaswagEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = HellaswagMultiChoiceDataModule(config)

        super().__init__(datamodule=datamodule, **kwargs)

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="test",
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
    ):
        outputs = super().evaluate(
            model,
            split=split,
            subsample=subsample,
            num_batches=num_batches,
            verbose=verbose,
            shuffle=shuffle,
        )
        return outputs["accuracy"]
