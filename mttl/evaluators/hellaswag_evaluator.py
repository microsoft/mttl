from mttl.datamodule.hellaswag_data_module import HellaswagMultiChoiceDataModule
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class HellaswagEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = HellaswagMultiChoiceDataModule(config)

        super().__init__(datamodule=datamodule, **kwargs)
