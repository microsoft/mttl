from src.data_transforms.templates import (
    QAPlatyInstructionGenerationTemplate,
    QAPlatyResponseGenerationTemplate,
)
from dataclasses import dataclass
import json


@dataclass
class QAModelSetting:
    inverse_model_path: str
    model_path: str
    instruction_template: str
    response_template: str

    @property
    def model_paths(self):
        return self.inverse_model_path, self.model_path


QA_MODEL_SETTINGS = {
    "platy": QAModelSetting(
        inverse_model_path="sordonia/llama2-13b-platypus-inverse",
        model_path="sordonia/llama2-13b-platypus",
        instruction_template=QAPlatyInstructionGenerationTemplate(),
        response_template=QAPlatyResponseGenerationTemplate(),
    )
}


class AutoConfig:
    @classmethod
    def from_path(cls, config_path):
        import json
        
        with open(config_path, "r") as f:
            config = json.load(f)

        type = config.pop("type")
        return eval(type)(**config)

    def save(self, config_path):
        config = self.__dict__
        config["type"] = self.__class__.__name__
        with open(config_path, "r") as f:
            json.dump(config, f)


@dataclass
class QATransformConfig(AutoConfig):
    model_setting: str
    max_context_length: int = 512
    max_tokens_instruction: int = 128
    max_tokens_response: int = 1024
    top_p: float = 0.9
    num_iterations: int = 1
    temperature: float = 0.7
    max_documents_per_subject: float = -1
    max_contexts_per_subject: float = -1


@dataclass
class QAForMMLUConfig(QATransformConfig):
    icl_examples: int = 0
    icl_dataset: str = "lukaemon/mmlu"
    icl_split: str = "validation"
    icl_use_options: str = True
