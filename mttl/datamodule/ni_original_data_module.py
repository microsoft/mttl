import numpy as np
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

import random
import string
from typing import Optional
import pkg_resources
from dataclasses import dataclass


from mttl.datamodule.utils import get_tokenizer
from mttl.datamodule.collators import DefaultCollator
from mttl.utils import hash_example, logger


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
            # add the input first.
            task_input += "Now complete the following example -\n"
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
                    definition = "Definition: " + instance["Definition"][0].strip()
                else:
                    definition = "Definition: " + instance["Definition"].strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                definition += "\n\n"

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
                pos_example_str += " " + self.tokenizer.eos_token
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
                if (
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
            if len(tokenized_source) <= self.max_input_length:
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

        # Randomly select one reference if multiple are provided.
        labels = [random.choice(ex["Instance"]["output"]) for ex in batch]
        # Add space for auto-regressive model tokenization
        labels = [" " + l for l in labels]

        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        task_names = [ex["Task"] for ex in batch]
        output_batch["task_names"] = task_names
        output_batch["task_ids"] = torch.LongTensor(
            [self.task_to_id[task] for task in task_names]
        )
        output_batch["labels_texts"] = labels
        output_batch["hashes"] = [hash_example(i + o) for i, o in zip(sources, labels)]
        output_batch["instruction_hashes"] = [hash_example(i) for i in sources]
        return output_batch


class NIOriginalDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def __init__(self, config, data_dir=None, for_generation=False):
        super().__init__()

        self.config = config
        self.dataset_reader = None
        self.data_dir = data_dir or config.data_dir
        self.for_generation = for_generation
        self.tokenizer = get_tokenizer(config)
        self.setup_dataset()

    def setup_dataset(self):
        filename = pkg_resources.resource_filename(
            __name__, "../dataloader/ni_original_dataset.py"
        )

        # if we are fine-tuning, we have to load ~1000 instances,
        # in order to be able to select the ones in training and in the valid set.
        dataset = load_dataset(
            filename,
            data_dir=self.data_dir,
            max_num_instances_per_task=(
                self.config.max_num_instances_per_task
                if self.config.finetune_task_name is None
                else self.config.max_num_instances_per_task * 2
            ),
        )

        if self.config.finetune_task_name is None:
            train_tasks = set(dataset["train"]["Task"])
            validation_tasks = set(dataset["validation"]["Task"])
            test_tasks = set(dataset["test"]["Task"])
            all_tasks = train_tasks.union(validation_tasks).union(test_tasks)
            self.task_to_id = {task: i for i, task in enumerate(all_tasks)}
            self.train_dataset = dataset["train"]
            self.val_dataset = dataset["validation"]
            self.test_dataset = dataset["test"]
        else:
            # take only the examples from the task we want to finetune on.
            self.task_to_id = {self.config.finetune_task_name: 0}
            train_indices = np.where(
                np.array(dataset["train"]["Task"]) == self.config.finetune_task_name
            )[0].tolist()
            valid_indices = np.where(
                np.array(dataset["validation"]["Task"])
                == self.config.finetune_task_name
            )[0].tolist()
            test_indices = np.where(
                np.array(dataset["test"]["Task"]) == self.config.finetune_task_name
            )[0].tolist()
            self.train_dataset = dataset["train"].select(train_indices)
            self.val_dataset = dataset["validation"].select(valid_indices)
            self.test_dataset = dataset["test"].select(test_indices)
            # we have to pick some examples from the validation set
            if len(self.train_dataset) == 0:
                self.train_dataset = self.val_dataset.select(
                    range(self.config.max_num_instances_per_task)
                )
                self.val_dataset = self.val_dataset.select(
                    range(
                        self.config.max_num_instances_per_task,
                        len(self.val_dataset),
                    )
                )

        self.collate_fn = DataCollatorForNI(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            num_pos_examples=self.config.num_pos_examples,
            add_task_definition=self.config.use_task_descriptions,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family=self.config.model_family
            if not self.for_generation
            else "seq2seq",
            task_to_id=self.task_to_id,
        )
        logger.info("Training examples: {}".format(len(self.train_dataset)))
        logger.info("Validation examples: {}".format(len(self.val_dataset)))
        logger.info("Test examples: {}".format(len(self.test_dataset)))


if __name__ == "__main__":
    from mttl.config import Config

    config = Config.parse()
    config.task_dir = "/datadrive2/sni/tasks"
    config.data_dir = "/datadrive2/sni/"
    config.model = "EleutherAI/gpt-neo-125m"
    datamodule = NIOriginalDataModule(config)
    datamodule.setup()
    print(next(iter(datamodule.train_dataloader())))
