from mttl.datamodule.humaneval_module import HumanEvalDataModule
from mttl.evaluators.code_evaluator import CodeEvaluator


class HumanEvalEvaluator(CodeEvaluator):
    def __init__(self, config, **kwargs):
        datamodule = HumanEvalDataModule(config, for_generation=True)

        super().__init__(
            datamodule=datamodule,
            prepend_source=not datamodule.config.use_instruct_template,
            **kwargs,
        )
