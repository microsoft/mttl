from mttl.datamodule.openbookqa_data_module import OpenbookQAMultiChoiceDataModule
from mttl.evaluators.base import switch_to_eval_mode
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class OpenbookQAEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = OpenbookQAMultiChoiceDataModule(config)

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
            model, split, subsample, num_batches, verbose, shuffle
        )
        return outputs["accuracy"]
