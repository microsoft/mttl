from mttl.datamodule.winogrande_data_module import WinograndeMultiChoiceDataModule
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class WinograndeEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = WinograndeMultiChoiceDataModule(config)

        super().__init__(datamodule=datamodule, **kwargs)
