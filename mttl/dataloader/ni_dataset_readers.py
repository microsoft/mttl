import hashlib
import json
import logging
import os
import string
from typing import Iterator, Tuple
import torch
import tqdm
import logging
import hashlib
import glob

from transformers import AutoTokenizer
from mttl.dataloader.ni_metrics import compute_ni_metrics
from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example


logger = logging.getLogger(__name__)


class NITaskDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        task_name,
        examples,
        tokenizer: AutoTokenizer,
        max_input_length,
        max_output_length,
        task_id=0,
        pretokenize=False,
    ):
        super().__init__()

        self.task_name = task_name
        self.task_id = task_id
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.pretokenize = pretokenize

        if pretokenize:
            self._pretokenize()

    def _pretokenize(self):
        inputs, outputs, instructions = zip(*self.examples)

        inputs_ids = self.tokenizer(
            list(inputs),
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        ).input_ids
        outputs_ids = self.tokenizer(
            list(outputs),
            truncation=True,
            padding="max_length",
            max_length=self.max_output_length,
            return_tensors="pt",
        ).input_ids

        input_hashes = [hash_example(i) for i in inputs]
        instruction_hashes = [hash_example(i) for i in instructions]

        self.examples = list(zip(
            inputs_ids,
            outputs_ids,
            input_hashes,
            instruction_hashes,
        ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        if self.pretokenize:
            tok_input, tok_output, input_hash, instruction_hash = self.examples[key]
        else:
            input, output, instruction = self.examples[key]
            tok_input = self.tokenizer(
                input,
                truncation=True,
                padding="max_length",
                max_length=self.max_input_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            tok_output = self.tokenizer(
                output,
                truncation=True,
                padding="max_length",
                max_length=self.max_output_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            input_hash = hash_example(input)
            instruction_hash = hash_example(instruction)

        ex_info = ExampleInfo(
            tok_input,
            tok_output,
            self.task_id,
            input_hash,
            example_id=key,
            input_text=input if not self.pretokenize else None,
            instruction_hash=instruction_hash,
        )
        return ex_info


class NIDatasetReader(object):
    # evaluation metric across tasks
    metric = "rougeL"

    @classmethod
    def load_data(cls, data_path, tasks=None):
        data = []
        tasks = tasks or [
            os.path.basename(t[:-5]) for t in list(glob.glob(data_path + "/*.json"))
        ]

        for task in tqdm.tqdm(tasks, desc=f"Loading {len(tasks)} tasks.."):
            with open(os.path.join(data_path, task + ".json"), "r") as f:
                data_ = json.load(f)
                # sanity check
                for example in data_["dev_examples"] + data_["test_examples"]:
                    assert len(example["Instance"]["output"]) == 1
                data.append(data_)
        return data

    @classmethod
    def format_example(
        cls,
        instance,
        tokenizer,
        use_task_descriptions=True,
        num_pos_examples=0,
        max_input_length=1024,
    ) -> Iterator[Tuple[str, str]]:
        """Format the input and iterate over all outputs corresponding to that input."""
        task_name = instance["Task"] + ". "

        task_input = cls._prepare_input(instance)

        if use_task_descriptions:
            definition = cls._prepare_definition(instance)
        else:
            definition = ""

        pos_examples = []
        if num_pos_examples > 0:
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
                pos_example_str += "\n"
                if (
                    len(
                        tokenizer(
                            definition
                            + " ".join(pos_examples)
                            + pos_example_str
                            + task_input
                        )["input_ids"]
                    )
                    <= max_input_length
                ):
                    pos_examples.append(pos_example_str)
                else:
                    break

        source = task_name + definition + "".join(pos_examples) + task_input
        tokenized_source = tokenizer(source)["input_ids"]

        # Trim input to max_input_length
        if len(tokenized_source) > max_input_length:
            source = tokenizer.decode(
                tokenized_source[:max_input_length], skip_special_tokens=True
            )

        task_information = definition

        # Select all references during training, this should only contain one reference during testing
        for output in instance["Instance"]["output"]:
            yield source, output, task_information

    def __init__(
        self,
        data_path,
        tokenizer,
        tasks,
        task2id=None,
        example2id=None,
        task_embed_path=None,
        max_input_length=1024,
        max_output_length=128,
        num_positive_examples=0,
        use_task_descriptions=True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.tasks = sorted(tasks)
        task_string = ",".join([self.data_path] + self.tasks).encode("utf8")
        self.tasks_hash = hashlib.md5(task_string).hexdigest()[:8]

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_task_descriptions = use_task_descriptions
        self.num_pos_examples = num_positive_examples

        self.task2id = task2id
        self.example2id = example2id
        self.task_embed_path = task_embed_path

        self.data = self.load_data(self.data_path, self.tasks)

    @classmethod
    def _prepare_input(cls, example):
        task_input = ""
        # add the input first.
        task_input += "Now complete the following example -\n"
        task_input += f"Input: {example['Instance']['input'].strip()}"
        if not task_input[-1] in string.punctuation:
            task_input += "."
        task_input += "\n"
        task_input += "Output: "
        return task_input

    @classmethod
    def _prepare_definition(cls, example):
        if isinstance(example["Definition"], list):
            definition = (
                "Definition: " + example["Definition"][0].strip()
            )
        else:
            definition = "Definition: " + example["Definition"].strip()
        if not definition[-1] in string.punctuation:
            definition += "."
        definition += "\n\n"
        return definition

    def read_all_instructions(self):
        """Read all instructions from the dataset.
        """
        all_instructions = []
        for data in self.data:
            # TODO: a bit of a lame loop. Definition is repeated inside each example.
            # yes I know it's a bit lame the preprocessing step should be changed in the future.
            all_examples = data["train_examples"] + data["dev_examples"] + data["test_examples"]
            if not all_examples:
                continue

            definition = self._prepare_definition(all_examples[0])
            all_instructions.append(definition)
        return all_instructions

    def read_orig_datasets(self, split):
        datasets = []

        for task in tqdm.tqdm(self.data):
            examples = []
            if split in ["train", "all"]:
                examples.extend(task["train_examples"])
            if split in ["dev", "val", "all"]:
                examples.extend(task["dev_examples"])
            if split in ["test"]:
                examples.extend(task["test_examples"])

            formatted_examples = []
            for example in examples:
                for formatted_example in self.format_example(
                    example,
                    self.tokenizer,
                    self.use_task_descriptions,
                    self.num_pos_examples,
                    self.max_input_length,
                ):
                    input, output, instruction = formatted_example
                    formatted_examples.append((input, output, instruction))

            # pretokenization takes a bit of time for NI
            task_dataset = NITaskDataset(
                task["task_name"],
                formatted_examples,
                self.tokenizer,
                self.max_input_length,
                self.max_output_length,
                task_id=self.task2id[task["task_name"]],
                pretokenize=False,
            )
            datasets.append(task_dataset)
        return datasets

    def decode(self, tokens):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    @property
    def test_examples(self):
        data = []
        for task_data in self.data:
            data.extend(task_data["test_examples"])
        return data

    def evaluate(self, predictions, split):
        predictions = [prediction.strip() for prediction in predictions]

        data = []
        for task_data in self.data:
            if split in ["train", "all"]:
                data.extend(task_data["train_examples"])
            if split in ["dev", "val", "all"]:
                data.extend(task_data["dev_examples"])
            if split in ["test"]:
                data.extend(task_data["test_examples"])

        if len(predictions) < len(data):
            raise ValueError("Not enough predictions for the data!")

        assert len(predictions) == len(data), (len(predictions), len(data))
        metric = compute_ni_metrics(
            predictions, data, pad_token_id=self.tokenizer.pad_token_id
        )
        return metric["exact_match"], metric["rougeL"]
