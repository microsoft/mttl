from mttl.datamodule.openbookqa_data_module import OpenbookQAMultiChoiceDataModule
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class OpenbookQAEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = OpenbookQAMultiChoiceDataModule(config)

        super().__init__(datamodule=datamodule, **kwargs)
