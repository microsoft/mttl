import numpy as np
import torch
import os
import pkg_resources
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Union, Optional

import re
import copy

from dataclasses import dataclass
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task

from mttl.datamodule.base import (
    DatasetConfig,
    DefaultCollator,
    DefaultDataModule,
)
from mttl.models.modifiers.expert_containers.expert_library import DatasetLibrary


#################################################
# Dataset aumgentation, implemented in the MMLU dataset
# Keeping these in case we want to augment in the data module
CHOICES = ["A", "B", "C", "D"]


def permute_options(dataset):
    def permute_options_fn(example):
        augmented_examples = copy.deepcopy(example)

        for i, instance in enumerate(example["Instance"]):
            input = instance["Input"]
            output = instance["Output"]
            output_idx = CHOICES.index(output)
            options = re.findall(r"\n[ABCD]\. .*", input)
            if len(options) > 4:
                continue
            for j in range(len(options)):
                new_options = options.copy()

                # put the correct answer in the jth position
                new_options[j], new_options[output_idx] = (
                    new_options[output_idx],
                    new_options[j],
                )

                # keep letter order
                new_options[j] = re.sub(
                    r"\n[ABCD]\.", f"\n{CHOICES[j]}.", new_options[j]
                )
                new_options[output_idx] = re.sub(
                    r"\n[ABCD]\.", f"\n{CHOICES[output_idx]}.", new_options[output_idx]
                )
                # new correct answer is at jth position
                new_output = CHOICES[j]
                assert re.sub(r"\n[ABCD]\.", "", options[output_idx]) == re.sub(
                    r"\n[ABCD]\.", "", new_options[j]
                ), "options not swapped correctly"
                # put options back into input
                _options = "".join(new_options) + "\nAnswer:"
                try:
                    to_replace = re.findall(
                        r"\n[ABCD]\. .*\nAnswer:", input, re.DOTALL
                    )[0]
                    new_input = input.replace(to_replace, _options)
                except:
                    pass

                augmented_examples["Task"].append(example["Task"][i])
                augmented_examples["Instance"].append(
                    {"Input": new_input, "Output": new_output}
                )
                augmented_examples["Positive Examples"].append(
                    example["Positive Examples"][i]
                )
                augmented_examples["Definition"].append(example["Definition"][i])

        return augmented_examples

    return dataset.map(permute_options_fn, batched=True)


def augment_prompts(dataset):
    """
    Inclusing prompts as discussed here: https://huggingface.co/blog/evaluating-mmlu-leaderboard
    Will only use AI Harness version of removing the prompt and adding Question: and Choices:
    """

    def _augment_prompts(example):
        augmented_examples = copy.deepcopy(example)
        for i, instance in enumerate(example["Instance"]):
            input = instance["Input"]

            if len(re.findall(r"\n[ABCD]\. .*", input)) > 4:
                continue

            input = "Question:\n" + input

            options = re.findall(r"\n[ABCD]\. .*\nAnswer:", input, re.DOTALL)

            input = input.replace(options[0], "\nChoices:\n" + options[0])

            augmented_examples["Instance"].append(
                {"Input": input, "Output": instance["Output"]}
            )
            augmented_examples["Positive Examples"].append(
                example["Positive Examples"][i]
            )
            augmented_examples["Definition"].append("")
            augmented_examples["Task"].append(example["Task"][i])

        return augmented_examples

    return dataset.map(_augment_prompts, batched=True)


@dataclass
class DataCollatorForMMLU(DefaultCollator):
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_input_length: Optional[int] = 2048
    max_output_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    for_generation: bool = False
    model_family: str = "seq2seq"
    task_to_id: dict = None
    few_shot: bool = True

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []

        for instance in batch:
            if self.few_shot:
                prompt = (
                    instance["Definition"]
                    + instance["Positive Examples"]
                    + instance["Instance"]["Input"]
                )
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

                while (
                    input_ids.shape[-1] > self.max_input_length
                    and len(instance["Positive Examples"].split("\n\n")) > 2
                ):
                    instance["Positive Examples"] = (
                        "\n\n".join(instance["Positive Examples"].split("\n\n")[:-2])
                        + "\n\n"
                    )
                    prompt = (
                        instance["Definition"]
                        + instance["Positive Examples"]
                        + instance["Instance"]["Input"]
                    )
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            else:
                prompt = instance["Definition"] + instance["Instance"]["Input"]

            sources.append(prompt)

        # Remove multiple spaces, which mess with tiktoken
        labels = [instance["Instance"]["Output"] for instance in batch]
        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        task_names = [instance["Task"] for instance in batch]
        output_batch["task_names"] = task_names

        if self.task_to_id is not None:
            output_batch["task_ids"] = torch.LongTensor(
                [self.task_to_id[task] for task in task_names]
            )

        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels
        return output_batch


@dataclass
class MMLUDataConfig(DatasetConfig):
    few_shot: bool = True
    augment_mmlu: bool = False


class MMLUDataModule(DefaultDataModule):
    DATA_ENV = "MMLU_DATA_DIR"

    def test_dataloader(self, subsample=None, shuffle=False):
        if subsample is not None and subsample > 0:
            from mttl.datamodule import take_n_examples_per_task

            indices = take_n_examples_per_task(
                list(self.test_dataset["Task"]),
                n=subsample,
                rng=self.rng if isinstance(self.rng, np.random.RandomState) else None,
            )
            test_dataset = self.test_dataset.select(indices)
        else:
            test_dataset = self.test_dataset

        return DataLoader(
            test_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def __init__(self, config: MMLUDataConfig, for_generation=False):
        if os.environ.get(self.DATA_ENV) is None:
            raise ValueError(
                f"Environment variable {self.DATA_ENV} is not set. "
                "Please set it to the directory containing the MMLU dataset."
            )

        super().__init__(config, for_generation=for_generation)

    @property
    def collate_fn(self):
        return DataCollatorForMMLU(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
            task_to_id=self.task_to_id,
            few_shot=self.config.few_shot,
        )

    def setup_dataset(self, stage=None):
        filename = pkg_resources.resource_filename(
            __name__, "../dataloader/mmlu_dataset.py"
        )
        dataset = DatasetLibrary.pull_dataset(
            filename,
            data_dir=os.environ[self.DATA_ENV],
            augment_with_prompts=self.config.augment_mmlu,
            augment_with_option_permutations=self.config.augment_mmlu,
        )

        (
            self._task_names,
            self._task_to_id,
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, "Task", self.config.finetune_task_name
        )
