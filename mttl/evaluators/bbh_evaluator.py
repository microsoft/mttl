from mttl.datamodule.bbh_data_module import BBHDataModule
from mttl.evaluators.base import GenerationOutput
from mttl.evaluators.em_evaluator import EMEvaluator


class DirectBBHEvaluator(EMEvaluator):
    def postprocess_generation_output(self, output: GenerationOutput):
        # only select the first part of the answer, and not the explanation etc.
        output.generated_texts = [
            text.split("\n\n")[0] for text in output.generated_texts
        ]
        return output

    def __init__(self, config, use_vllm=False, generation_kwargs=None):
        datamodule = BBHDataModule(config, for_generation=True)

        generation_kwargs["max_new_tokens"] = 20
        super().__init__(
            datamodule=datamodule,
            use_vllm=use_vllm,
            generation_kwargs=generation_kwargs,
        )
