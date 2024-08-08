from mttl.datamodule.arc_data_module import ArcDataConfig, ArcMultiChoiceDataModule
from mttl.datamodule.codex_data_module import CodexDataConfig, CodexDataModule
from mttl.datamodule.hellaswag_data_module import (
    HellaswagDataConfig,
    HellaswagMultiChoiceDataModule,
)
from mttl.datamodule.mmlu_data_module import MMLUDataConfig, MMLUDataModule
from mttl.datamodule.mt_seq_to_seq_module import (
    FlanConfig,
    FlanModule,
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)
from mttl.datamodule.openbookqa_data_module import (
    OpenbookQADataConfig,
    OpenbookQAMultiChoiceDataModule,
)
from mttl.datamodule.piqa_data_module import PiqaDataConfig, PiqaMultiChoiceDataModule
from mttl.datamodule.superglue_data_module import BoolQDataModule, SuperGLUEDataConfig
from mttl.datamodule.winogrande_data_module import (
    WinograndeDataConfig,
    WinograndeMultiChoiceDataModule,
)
