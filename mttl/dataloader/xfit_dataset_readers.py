import hashlib
import logging
import os
from typing import Iterator, Tuple
import torch
import tqdm
import numpy as np
import logging
import hashlib

from mttl.dataloader.xfit_task_descriptions import DESCRIPTIONS
from mttl.dataloader.xfit_metrics import METRICS
from mttl.dataloader.data_utils import ExampleInfo
from mttl.utils import hash_example


logger = logging.getLogger(__name__)


class XFitTaskDataset(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        task_name,
        examples,
        tokenizer,
        max_input_length,
        max_output_length,
        task_id=0,
        pretokenize=True,
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
        inputs, outputs, hashes = zip(*self.examples)

        # flatten outputs for tokenization
        flat_outputs = []
        for output in outputs:
            flat_outputs.extend(output)

        inputs = self.tokenizer(
            list(inputs),
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        flat_outputs = self.tokenizer(
            flat_outputs,
            truncation=True,
            padding="max_length",
            max_length=self.max_output_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # unflatten outputs
        i = 0
        outputs_ = []
        for output in outputs:
            outputs_.append(flat_outputs[i : i + len(output)])
            i += len(output)

        self.examples = list(zip(inputs, outputs_, hashes))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, key):
        input, outputs, hash = self.examples[key]
        output_idx = np.random.randint(len(outputs))

        if self.pretokenize:
            tok_input, tok_output = (input, outputs[output_idx])
        else:
            tok_input = self.tokenizer(
                input,
                truncation=True,
                padding="max_length",
                max_length=self.max_input_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            tok_output = self.tokenizer(
                outputs[output_idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_output_length,
                return_tensors="pt",
            ).input_ids.squeeze(0)

        return ExampleInfo(
            tok_input, tok_output, self.task_id, hash, example_id=key,
        )


class XFitDatasetReader(object):
    # evaluation metric across tasks
    general_metric = "EM"

    @classmethod
    def load_examples(cls, file_path):
        with open(file_path) as fin:
            lines = fin.readlines()

        examples = []
        for line in lines:
            d = line.strip().split("\t")
            examples.append((d[0], d[1:]))
        return examples

    @classmethod
    def load_data(cls, data_path, tasks, task_prefix=None):
        data = []

        # keep "sorted" so that things are consistent across machines
        for _, task in enumerate(sorted(tasks)):
            task_dir = os.path.join(data_path, task)
            files = sorted(os.listdir(task_dir))
            prefixes = []

            for filename in files:
                if not filename.endswith(".tsv"):
                    continue

                prefix = "_".join(filename.split("_")[:-1])

                if prefix not in prefixes:
                    if task_prefix and prefix != task_prefix:
                        continue

                    prefixes.append(prefix)

            for prefix in prefixes:
                train_examples = cls.load_examples(
                    os.path.join(task_dir, prefix + "_train.tsv")
                )
                dev_examples = cls.load_examples(
                    os.path.join(task_dir, prefix + "_dev.tsv")
                )
                test_examples = cls.load_examples(
                    os.path.join(task_dir, prefix + "_test.tsv")
                )

                data.append(
                    {
                        "task_name": task,
                        "task_prefix": prefix,
                        "train_examples": train_examples,
                        "dev_examples": dev_examples,
                        "test_examples": test_examples,
                    }
                )
        return data

    @classmethod
    def format_example(
        cls,
        instance,
        task_name,
        do_lowercase=False,
        append_another_bos=False,
        use_task_descriptions=False,
    ) -> Iterator[Tuple[str, str]]:

        input, outputs = instance
        if use_task_descriptions:
            description = DESCRIPTIONS[task_name]
            input = description + ". " + input

        input = " [{}] {}".format(task_name, input)
        outputs = [" " + item for item in outputs]

        if do_lowercase:
            input = input.lower()
            outputs = [output.lower() for output in outputs]

        if append_another_bos:
            input = "<s> " + input
            outputs = ["<s> " + output for output in outputs]

        return input, outputs, hash_example(input)

    def __init__(
        self,
        data_path,
        tokenizer,
        tasks,
        task_prefix=None,
        task2id=None,
        example2id=None,
        task_embed_path=None,
        max_input_length=512,
        max_output_length=64,
        use_task_descriptions=False,
        do_lowercase=False,
        append_another_bos=False,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.tasks = sorted(tasks)
        self.task_prefix = task_prefix
        task_string = ",".join([self.data_path] + self.tasks).encode("utf8")
        self.tasks_hash = hashlib.md5(task_string).hexdigest()[:8]

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.use_task_descriptions = use_task_descriptions
        self.do_lowercase = do_lowercase
        self.append_another_bos = append_another_bos

        self.task2id = task2id
        self.example2id = example2id
        self.task_embed_path = task_embed_path

        self.data = self.load_data(self.data_path, self.tasks, task_prefix=task_prefix)

        if len(self.data) > 1:
            self.metric = self.general_metric
        else:
            self.metric = METRICS[self.data[0]["task_name"]]

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
                formatted_example = self.format_example(
                    example,
                    task["task_name"],
                    do_lowercase=self.do_lowercase,
                    append_another_bos=self.append_another_bos,
                    use_task_descriptions=self.use_task_descriptions,
                )
                formatted_examples.append(formatted_example)

            task_dataset = XFitTaskDataset(
                task["task_name"],
                formatted_examples,
                self.tokenizer,
                self.max_input_length,
                self.max_output_length,
                self.task2id[task["task_name"]],
                pretokenize=True,
            )
            datasets.append(task_dataset)
        return datasets

    @property
    def test_examples(self):
        data = []
        for task_data in self.data:
            data.extend(task_data["test_examples"])
        return [{"inputs": d[0], "outputs": d[1]} for d in data]

    def decode(self, tokens):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def decode_batch(self, tokens):
        return [self.decode(token) for token in tokens]

    def evaluate(self, predictions, split):
        from .xfit_metrics import evaluate

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
            logger.warning(
                "not enough predictions were given. truncating target sentences from "
                + f"{len(self)} to {len(predictions)}"
            )
            data = data[: len(predictions)]

        assert len(predictions) == len(data), (len(predictions), len(data))

        # general metric
        general_metric = task_metric = evaluate(predictions, data, self.general_metric)

        # only one task is loaded
        if len(self.data) == 1:
            task_metric = evaluate(
                predictions, data, METRICS[self.data[0]["task_name"]]
            )

        return general_metric, task_metric

    def save_predictions(self, predictions, output_dir):
        assert len(predictions) == len(self), (len(predictions), len(self))

        predictions = [
            "n/a" if len(prediction.strip()) == 0 else prediction
            for prediction in predictions
        ]
        prediction_text = [prediction.strip() + "\n" for prediction in predictions]
        save_path = os.path.join(
            output_dir, "predictions_{}.txt".format(self.task_name)
        )
        with open(save_path, "w") as f:
            f.writelines(prediction_text)

        logger.info("Saved prediction in {}".format(save_path))
