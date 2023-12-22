from mttl.datamodule.bbh_data_module import BBHDataModule
from mttl.evaluators.em_evaluator import EMEvaluator


class DirectBBHEvaluator(EMEvaluator):
    def __init__(self, config, use_vllm=False, generation_kwargs=None):
        self.datamodule = BBHDataModule(config)

        generation_kwargs["max_new_tokens"] = 1
        super().__init__(
            config=config,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )
