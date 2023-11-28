from abc import abstractmethod
from typing import Dict, List
import json


class TransformConfig:
    @classmethod
    def from_path(cls, config_path):
        from projects.wiki_experts.src.data_transforms.qa import (
            QATransformConfig,
        )  # noqa
        from projects.wiki_experts.src.data_transforms.facts import (
            FactsTransformConfig,
        )  # noqa
        from projects.wiki_experts.src.data_transforms.facts import (
            IDTransformConfig,
        )  # noqa
        from projects.wiki_experts.src.data_transforms.description import (
            DescTransformConfig,
        )  # noqa
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


class DataTransformTemplate:
    @classmethod
    def apply(cls, *args, **kwargs) -> str:
        pass

    @classmethod
    def post_process_generation(cls, output):
        pass


class TransformModel:
    icl_sampler = None

    @abstractmethod
    def transform(self, dataset_name, **options) -> List[Dict]:
        pass

    @classmethod
    def from_config(cls, transform_config: TransformConfig):
        from projects.wiki_experts.src.data_transforms.qa import (
            MMLUICLSampler,
            QATransformModel,
            QATransformConfig,
        )
        from projects.wiki_experts.src.data_transforms.facts import (
            FactsTransformConfig,
            FactsTransformModel,
            IDTransformConfig,
            IDTransformModel,
        )
        from projects.wiki_experts.src.data_transforms.description import (
            DescTransformModel,
            DescTransformConfig,
        )

        if type(transform_config) == QATransformConfig:
            if transform_config.icl_examples > 0:
                cls.icl_sampler = MMLUICLSampler(
                    transform_config.icl_dataset,
                    transform_config.icl_split,
                    transform_config.icl_use_options,
                )
            return QATransformModel(transform_config)
        elif type(transform_config) == FactsTransformConfig:
            return FactsTransformModel(transform_config)
        elif type(transform_config) == IDTransformConfig:
            return IDTransformModel(transform_config)
        elif type(transform_config) == DescTransformConfig:
            return DescTransformModel(transform_config)
