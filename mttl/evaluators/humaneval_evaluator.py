from mttl.datamodule.humaneval_module import HumanEvalDataModule
from mttl.evaluators.code_evaluator import CodeEvaluator


class HumanEvalEvaluator(CodeEvaluator):
    STOP_TOKENS = ["\n\n", "\ndef"]

    def __init__(self, config, generation_kwargs=None, **kwargs):
        datamodule = HumanEvalDataModule(config, for_generation=True)

        generation_kwargs = generation_kwargs or {}
        generation_kwargs.update({"stop_tokens": self.STOP_TOKENS})

        super().__init__(
            datamodule=datamodule,
            prepend_source=not datamodule.config.use_instruct_template,
            **kwargs,
        )
