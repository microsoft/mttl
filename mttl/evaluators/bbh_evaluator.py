from mttl.datamodule.bbh_data_module import BBHDataModule
from mttl.evaluators.base import GenerationOutput
from mttl.evaluators.em_evaluator import EMEvaluator


class DirectBBHEvaluator(EMEvaluator):
    def __init__(self, config, use_vllm=False, generation_kwargs=None):
        datamodule = BBHDataModule(config, for_generation=True)

        generation_kwargs["stop_tokens"] = ["\n\n"]

        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )

    def evaluate(self, *args, **kwargs):
        """EM returns a score between 0 and 1, so we divide by 100 to get a score between 0 and 1."""
        return super().evaluate(*args, **kwargs) / 100.0


class DirectBBHEvaluatorFast(DirectBBHEvaluator):
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs, num_batches=200, shuffle=True)
