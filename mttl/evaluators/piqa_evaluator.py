from mttl.datamodule.piqa_data_module import PiqaMultiChoiceDataModule
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class PiqaEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = PiqaMultiChoiceDataModule(config)

        super().__init__(datamodule=datamodule, **kwargs)
