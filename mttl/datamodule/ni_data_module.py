import json
import os
import random
import string
from dataclasses import dataclass
from importlib.resources import files
from typing import Optional

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.logging import logger
from mttl.models.library.expert_library import DatasetLibrary


@dataclass
class NiDataConfig(DatasetConfig):
    add_task_name: bool = False
    add_task_definition: bool = True
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    sep_by_eos_token: bool = False
    tk_instruct: bool = False
    max_num_instances_per_task: int = 100


@dataclass
class DataCollatorForNI(DefaultCollator):
    tokenizer: AutoTokenizer
    padding: bool = True
    max_input_length: Optional[int] = 1024
    max_output_length: Optional[int] = 128
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    add_task_name: bool = False
    add_task_definition: bool = True
    sep_by_eos_token: bool = False
    num_pos_examples: int = 0
    num_neg_examples: int = 0
    add_explanation: bool = False
    tk_instruct: bool = False
    model_family: str = None
    task_to_id: dict = None

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        sources = []
        for instance in batch:
            if self.tk_instruct:
                all_valid_encodings = [
                    # instruction only
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 0,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # example only
                    {
                        "add_task_name": False,
                        "add_task_definition": False,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": False,
                    },
                    # instruction + pos examples + neg examples
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 2,
                        "add_explanation": False,
                    },
                    # instruction + pos (w. explanation)
                    {
                        "add_task_name": False,
                        "add_task_definition": True,
                        "num_pos_examples": 2,
                        "num_neg_examples": 0,
                        "add_explanation": True,
                    },
                ]
                encoding_schema = random.choice(all_valid_encodings)
                add_task_name = encoding_schema["add_task_name"]
                add_task_definition = encoding_schema["add_task_definition"]
                num_pos_examples = encoding_schema["num_pos_examples"]
                num_neg_examples = encoding_schema["num_neg_examples"]
                add_explanation = encoding_schema["add_explanation"]
            else:
                add_task_name = self.add_task_name
                add_task_definition = self.add_task_definition
                num_pos_examples = self.num_pos_examples
                num_neg_examples = self.num_neg_examples
                add_explanation = self.add_explanation

            task_input = ""
            task_input += "Now complete the following example -\n"
            # add the input first.
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output:"

            task_name = ""
            if add_task_name:
                task_name += instance["Task"] + ". "

            definition = ""
            if add_task_definition:
                if isinstance(instance["Definition"], list):
                    definition = "Task definition: " + instance["Definition"][0].strip()
                else:
                    definition = "Task definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n"

            # try to add positive examples.
            pos_examples = []
            for idx, pos_example in enumerate(
                instance["Positive Examples"][:num_pos_examples]
            ):
                pos_example_str = f" Positive Example {idx+1} -\n"
                pos_example_str += f"Input: {pos_example['input'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                pos_example_str += "\n"
                pos_example_str += f" Output: {pos_example['output'].strip()}"
                if not pos_example_str[-1] in string.punctuation:
                    pos_example_str += "."
                # add eos token
                pos_example_str += (
                    (" " + self.tokenizer.eos_token) if self.sep_by_eos_token else ""
                )
                # end add eos token
                pos_example_str += "\n"
                if add_explanation and "explanation" in pos_example:
                    pos_example_str += (
                        f" Explanation: {pos_example['explanation'].strip()}"
                    )
                    if not pos_example_str[-1] in string.punctuation:
                        pos_example_str += "."
                    pos_example_str += "\n"
                pos_example_str += "\n"
                if self.max_input_length < 0 or (
                    len(
                        self.tokenizer(
                            definition
                            + " ".join(pos_examples)
                            + pos_example_str
                            + task_input
                        )["input_ids"]
                    )
                    <= self.max_input_length
                ):
                    pos_examples.append(pos_example_str)
                else:
                    break

            # try to add negative examples.
            neg_examples = []
            for idx, neg_example in enumerate(
                instance["Negative Examples"][:num_neg_examples]
            ):
                neg_example_str = f" Negative Example {idx+1} -\n"
                neg_example_str += f"Input: {neg_example['input'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                neg_example_str += "\n"
                neg_example_str += f" Output: {neg_example['output'].strip()}"
                if not neg_example_str[-1] in string.punctuation:
                    neg_example_str += "."
                # add eos token
                neg_example_str += " " + self.tokenizer.eos_token
                # end add eos token
                neg_example_str += "\n"
                if add_explanation and "explanation" in neg_example:
                    neg_example_str += (
                        f" Explanation: {neg_example['explanation'].strip()}"
                    )
                    if not neg_example_str[-1] in string.punctuation:
                        neg_example_str += "."
                    neg_example_str += "\n"
                neg_example_str += "\n"
                if (
                    len(
                        self.tokenizer(
                            definition
                            + " ".join(pos_examples)
                            + " ".join(neg_examples)
                            + neg_example_str
                            + task_input
                        )["input_ids"]
                    )
                    <= self.max_input_length
                ):
                    neg_examples.append(neg_example_str)
                else:
                    break

            source = (
                task_name
                + definition
                + "".join(pos_examples)
                + "".join(neg_examples)
                + task_input
            )
            tokenized_source = self.tokenizer(source)["input_ids"]
            if (
                len(tokenized_source) <= self.max_input_length
                or self.max_input_length < 0
            ):
                sources.append(source)
            else:
                tokenized_task_input = self.tokenizer(
                    "\nOutput:", add_special_tokens=False
                )["input_ids"]
                sources.append(
                    self.tokenizer.decode(
                        tokenized_source[
                            : self.max_input_length - len(tokenized_task_input)
                        ],
                        skip_special_tokens=True,
                    )
                    + "\nOutput:"
                )

        output_batch = {}

        labels_full_seq = [ex["Instance"]["output"] for ex in batch]
        labels_rand = [random.choice(ex["Instance"]["output"]) for ex in batch]
        instance_ids = [ex["Instance"]["id"] for ex in batch]
        task_categories = [ex["Categories"] for ex in batch]
        task_identifiers = [ex["Task"] for ex in batch]

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels_rand)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels_rand)
        )

        output_batch["task_names"] = task_identifiers
        output_batch["task_identifiers"] = (
            task_identifiers  # sni task id like e.g. task1356_xlsum_title_generation
        )
        output_batch["task_categories"] = task_categories
        output_batch["task_ids"] = torch.LongTensor(
            [self.task_to_id[task] for task in task_identifiers]
        )  # task ids potentially used in routing
        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels_rand
        output_batch["labels_full_seq"] = labels_full_seq
        output_batch["instance_ids"] = instance_ids
        return output_batch


@DataModule.register("ni", config_cls=NiDataConfig)
class NiDataModule(DataModule):
    def test_dataloader(self, subsample=-1, shuffle=False):
        if subsample > 0:
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
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def __init__(self, config, for_generation=False, val_mixin=None):
        if "NI_DATA_DIR" not in os.environ:
            raise ValueError("Please set the NI_DATA_DIR environment variable.")

        super().__init__(config, for_generation=for_generation, val_mixin=val_mixin)

    def _check_test_references(self):
        # make sure all test instances are in reference file
        reference_file = os.path.join(
            os.environ["NI_DATA_DIR"], "test_references.jsonl"
        )

        if not os.path.exists(reference_file):
            logger.warning("No test references found, skipping check.")
            return

        eval_instances = {}
        with open(reference_file) as fin:
            for line in fin:
                instance = json.loads(line)
                # if track is not provided in the refernce file, we use set the track to `default` and use the default tokenizer in rouge-score.
                if "track" not in instance:
                    instance["track"] = "default"
                eval_instances[instance["id"]] = instance

        eval_ids = list(eval_instances.keys())
        for element in tqdm.tqdm(
            self.test_dataset,
            desc="Checking test instances",
            total=len(self.test_dataset),
        ):
            id = element["id"]
            assert (
                id in eval_ids
            ), f"{id} not in test references, see https://github.com/allenai/natural-instructions/blob/master/eval/leaderboard/create_reference_file.py"

    @property
    def collate_fn(self):
        return DataCollatorForNI(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            num_pos_examples=self.config.num_pos_examples,
            add_task_definition=self.config.add_task_definition,
            sep_by_eos_token=self.config.sep_by_eos_token,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
            task_to_id=self.task_to_id,
        )

    def setup_dataset(self):
        filename = str(files("mttl.dataloader").joinpath("ni_dataset.py"))

        # if we are fine-tuning, we have to load ~1000 instances,
        # in order to be able to select the ones in training and in the valid set.
        dataset = DatasetLibrary.pull_dataset(
            filename,
            data_dir=os.environ["NI_DATA_DIR"],
            task_name=self.config.finetune_task_name,
            max_num_instances_per_task=(
                self.config.max_num_instances_per_task
                if self.config.finetune_task_name is None
                else self.config.max_num_instances_per_task * 2
            ),
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

        # we have to pick some examples from the validation set
        if len(self.train_dataset) == 0:
            self.train_dataset = self.dev_dataset.select(
                range(self.config.max_num_instances_per_task)
            )
            self.dev_dataset = self.dev_dataset.select(
                range(
                    self.config.max_num_instances_per_task,
                    len(self.dev_dataset),
                )
            )

        self._check_test_references()
