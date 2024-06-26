from mttl.datamodule.superglue_data_module import BoolQDataModule
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class BoolQEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = BoolQDataModule(config)

        super().__init__(datamodule=datamodule, **kwargs)
