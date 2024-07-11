import os
import sys

from mttl.datamodule.hellaswag_data_module import (
    HellaswagDataConfig,
    HellaswagMultiChoiceDataModule,
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

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.datamodule.arc_data_module import ArcDataConfig, ArcMultiChoiceDataModule
from mttl.datamodule.codex_data_module import CodexDataConfig, CodexDataModule
from mttl.datamodule.mmlu_data_module import MMLUDataConfig, MMLUDataModule
from mttl.datamodule.mt_seq_to_seq_module import (
    FlanConfig,
    FlanModule,
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)


def get_datamodule(args, for_generation=False, dataset_override=None):
    # refactor all the common arguments below into a dict common kwargs
    dataset = args.dataset if not dataset_override else dataset_override

    common_kwargs = {
        "model": args.model,
        "train_batch_size": args.train_batch_size,
        "predict_batch_size": args.predict_batch_size,
        "max_input_length": args.max_input_length,
        "max_output_length": args.max_output_length,
        "validation_portion": args.validation_portion,
        "model_family": args.model_family,
        "finetune_task_name": args.finetune_task_name,
        "truncation_side": args.truncation_side,
        "dataset": dataset,
        "train_on_inputs": False,
        "add_eos_to_targets": "qamc"
        not in args.dataset,  # do not add eos for mmlu stuff (for now)
        "subsample_train": args.subsample_train,
        "subsample_dev": args.subsample_dev,
        "subsample_test": args.subsample_test,
    }
    if dataset in [
        "arc-easy",
        "arc-challenge",
        "arc_easy",
        "arc_challenge",
        "openbookqa",
        "boolq",
        "piqa",
        "winogrande",
        "hellaswag",
    ]:
        dataset_to_klass_map = {
            "arc-easy": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Easy"),
                ArcMultiChoiceDataModule,
            ),
            "arc_easy": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Easy"),
                ArcMultiChoiceDataModule,
            ),
            "arc-challenge": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Challenge"),
                ArcMultiChoiceDataModule,
            ),
            "arc_challenge": (
                ArcDataConfig(**common_kwargs, arc_type="ARC-Challenge"),
                ArcMultiChoiceDataModule,
            ),
            "openbookqa": (
                OpenbookQADataConfig(**common_kwargs),
                OpenbookQAMultiChoiceDataModule,
            ),
            "boolq": (SuperGLUEDataConfig(**common_kwargs), BoolQDataModule),
            "piqa": (PiqaDataConfig(**common_kwargs), PiqaMultiChoiceDataModule),
            "winogrande": (
                WinograndeDataConfig(**common_kwargs),
                WinograndeMultiChoiceDataModule,
            ),
            "hellaswag": (
                HellaswagDataConfig(**common_kwargs),
                HellaswagMultiChoiceDataModule,
            ),
        }
        assert not for_generation
        config = dataset_to_klass_map[dataset][0]
        dm = dataset_to_klass_map[dataset][1](config)
    elif "flan" in dataset:
        config = FlanConfig(
            **common_kwargs,
            remove_phi_eval_tasks=args.remove_phi_eval_tasks,
            include_task_source=args.include_task_source,
        )
        dm = FlanModule(config, for_generation=for_generation)
    elif "flat" in dataset:
        config = FlatMultiTaskConfig(
            **common_kwargs,
            source_template=args.source_template,
            augment_few_shot=args.augment_few_shot,
        )
        dm = FlatMultiTaskModule(config, for_generation=for_generation)
    elif "mmlu" in dataset:
        config = MMLUDataConfig(
            **common_kwargs,
        )
        dm = MMLUDataModule(config, for_generation=for_generation)
    elif "codex" in dataset:
        config = CodexDataConfig(
            **common_kwargs,
        )
        dm = CodexDataModule(config, for_generation=for_generation)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return dm
