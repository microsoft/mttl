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


class DirectBBHEvaluatorFast(DirectBBHEvaluator):
    def evaluate(*args, **kwargs):
        super().evaluate(*args, **kwargs, num_batches=200)
