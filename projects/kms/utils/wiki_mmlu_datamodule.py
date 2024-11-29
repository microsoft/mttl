import json
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, MultiChoiceDataModule
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task, split_on_split_column
from projects.kms.utils.quality_datamodule import QualityDatamodule


@dataclass
class WikiMMLUDatasetConfig(DatasetConfig):
    task_name_field: str = "document_id"
    task_source_field: str = "document_id"
    prompt: str = (
        "Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness. Question: "
    )
    include_context: bool = False
    topk_context: int = 10


@DataModule.register("wiki_mmlu", config_cls=WikiMMLUDatasetConfig)
class WikiMMLUDataModule(QualityDatamodule):
    pass
