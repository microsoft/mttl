from abc import abstractmethod
from typing import Dict, List

from projects.wiki_experts.src.data_transforms.qa import MMLUICLSampler, QATransformModel
from src.data_transforms.config import (
    QATransformConfig,
    AutoConfig,
)

class AutoTransform:
    icl_sampler = None

    @abstractmethod
    def transform(self, dataset_name, **options) -> List[Dict]:
        pass

    @classmethod
    def from_config(cls, transform_config: AutoConfig):
        if type(transform_config) == QATransformConfig:
            if transform_config.icl_examples > 0:
                cls.icl_sampler = MMLUICLSampler(
                    transform_config.icl_dataset,
                    transform_config.icl_split,
                    transform_config.icl_use_options
                )
            return QATransformModel(transform_config)
