import math
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import get_worker_info

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task, split_on_split_column
from mttl.logging import logger
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class PITDatasetConfig(DatasetConfig):
    # for each input example, we could have several outputs (e.g. several summaries or QA pairs), we can only use N of these
    num_outputs_per_chunk: int = -1
    # field in the dataset that contains the task name
    task_name_field: str = "document_id"
    # field in the dataset that contains the task source
    task_source_field: str = "document_id"
    # for train / dev split, split by document chunk, or split the list of summary / q/a's ?
    split_train_dev_on: str = "document_chunk"
    # there might be multiple types, i.e. "qa", "summary", or maybe else in the future
    use_only_type: str = "qa"


class PITDataCollator(DefaultCollator):
    pass


@DataModule.register("pit", config_cls=PITDatasetConfig)
class PITDatasetModule(DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.config.split_train_dev_on in ["document_chunk", "output"]

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        assert "train" in dataset

        def filter_targets(example, n):
            return {"outputs": example["outputs"][:n]}

        if self.config.num_outputs_per_chunk > 0:
            logger.info(
                f"Keeping only {self.config.num_outputs_per_chunk} outputs per document."
            )
            dataset = dataset.map(
                partial(filter_targets, n=self.config.num_outputs_per_chunk),
                num_proc=20,
            )

        def expand_targets_and_chat(example):
            return_dict = defaultdict(list)

            valid_types = set(self.config.use_only_type.split(","))
            assert valid_types.issubset({"summary", "qa"})

            for i in range(len(example["input"])):
                input = example["input"][i]
                outputs = example["outputs"][i]
                type_ = example["type"][i]
                subject = example[self.config.task_name_field][i]

                # TODO : make compatible for Q/A as well!
                if type_ not in valid_types:
                    continue

                for idx, output in enumerate(outputs):
                    # for QA, we want to show the question as well as the prompt
                    if type_ == "qa":
                        if isinstance(output, dict):
                            qa = f"\nQuestion: {output['question']}\nAnswer: {output['answer']}"
                        elif isinstance(output, str):
                            qa = output
                        output_str = qa + "\nDocument:\n" + input
                        prompt_str = "Generate questions and answers and the corresponding full paragraph."

                    elif type_ == "summary":
                        if type(output) == dict:
                            summary = output["summary"]
                        else:
                            summary = output

                        prompt_str = (
                            "Generate a summary and the corresponding full paragraph."
                        )
                        output_str = "Summary:\n" + summary + "\nDocument:\n" + input

                    source_str = self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": prompt_str,
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    return_dict["source"].append(source_str)
                    return_dict["target"].append(output_str)

                    if self.config.split_train_dev_on == "output":
                        # ensure that 95% or at least 1 point goes to dev
                        if idx < max(int(len(outputs) * 0.05), 1):
                            return_dict["split"].append("dev")
                        else:
                            return_dict["split"].append("train")

                    return_dict[self.config.task_name_field].append(subject)
            return return_dict

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset,
            self.config.task_name_field,
            self.config.finetune_task_name,
            n_proc=n_proc,
        )

        train_dataset = train_dataset.map(
            expand_targets_and_chat,
            batched=True,
            batch_size=1000,
            desc="Applying chat template...",
            remove_columns=train_dataset.column_names,
        )

        if self.config.split_train_dev_on == "document_chunk":
            self.train_dataset, self.dev_dataset = self.create_train_valid_split(
                train_dataset
            )
        else:
            # split on `split` column
            self.train_dataset, self.dev_dataset, _ = split_on_split_column(
                train_dataset
            )

        self.test_dataset = self.dev_dataset

        logger.info("PIT Datamodule set-up!")
        logger.info(f"Example source: {self.train_dataset[0]['source']}")
        logger.info(f"Example target: {self.train_dataset[0]['target']}")
