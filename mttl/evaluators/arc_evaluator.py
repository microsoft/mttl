from mttl.datamodule.arc_data_module import ArcMultiChoiceDataModule
from mttl.evaluators.loglike_evaluator import LogLikeEvaluator


class ArcEvaluator(LogLikeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = ArcMultiChoiceDataModule(config)

        super().__init__(datamodule=datamodule, **kwargs)
