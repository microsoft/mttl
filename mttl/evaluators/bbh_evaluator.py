from mttl.datamodule.bbh_data_module import BBHDataModule
from mttl.evaluators.em_evaluator import EMEvaluator


class BBHEvaluator(EMEvaluator):
    def __init__(self, config, use_vllm=False, generation_kwargs=None):
        self.datamodule = BBHDataModule(config)

        super().__init__(
            config=config,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )
