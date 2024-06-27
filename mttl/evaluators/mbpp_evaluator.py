from mttl.datamodule.mbpp_datamodule import MBPPDataModule
from mttl.evaluators.code_evaluator import CodeEvaluator


class MBPPEvaluator(CodeEvaluator):
    STOP_TOKENS = ["\n\n", "\ndef"]

    def __init__(self, config, use_vllm=False, generation_kwargs=None, split="test"):
        datamodule = MBPPDataModule(config, for_generation=True)

        generation_kwargs = generation_kwargs or {}
        generation_kwargs.update({"stop_tokens": self.STOP_TOKENS})

        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
            prepend_source=not datamodule.config.use_instruct_template,
            split=split,
        )
