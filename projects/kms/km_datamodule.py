import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.logging import logger
from mttl.models.library.dataset_library import DatasetLibrary

AVAILABLE_PROMPTS = {
    "summary": "Summarize the preceding passage.",
    "qa": "Generate a question-answer pair given the preceding passage.",
}


@dataclass
class KMDatasetConfig(DatasetConfig):
    # there might be multiple types, i.e. "qa", "summary", or maybe else in the future
    use_only_type: str = None
    # for each input example, we could have several outputs (e.g. several summaries or QA pairs), we can only use N of these
    num_outputs_per_chunk: int = 4
    # field in the dataset that contains the task name
    task_name_field: str = "subject"
    # field in the dataset that contains the task source
    task_source_field: str = "subject"


class KMDataCollator(DefaultCollator):
    def __call__(self, batch):
        if "input_ids" in batch[0]:
            return self.packed_collate(batch)

        output_batch = super().__call__(batch)

        prompts = [b["prompt"] for b in batch]
        labels = [b["target"] for b in batch]
        prompt_batch = self.prepare_inputs_for_gpt_family(prompts, labels)

        # no context tensors used for context distillation loss
        output_batch["nc_input_ids"] = prompt_batch["input_ids"]
        output_batch["nc_attention_mask"] = prompt_batch["attention_mask"]
        output_batch["nc_labels"] = prompt_batch["labels"]

        return output_batch


@DataModule.register("dcd_km", config_cls=KMDatasetConfig)
class KMDatasetModule(DataModule):
    collate_class = KMDataCollator

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        assert "train" in dataset

        def filter_targets(example, n):
            return {"outputs": example["outputs"][:n]}

        logger.info(
            f"Keeping only {self.config.num_outputs_per_chunk} outputs per document."
        )
        dataset = dataset.map(
            partial(filter_targets, n=self.config.num_outputs_per_chunk), num_proc=20
        )

        if self.config.use_only_type:
            # filter types (e.g. use only summary, or use only qas)
            def filter_types(example, types):
                return example["type"] in types.split(",")

            dataset = dataset.filter(
                partial(filter_types, type=self.config.use_only_type), num_proc=20
            )

        def expand_targets_and_chat(example):
            return_dict = {
                "source": [],
                "target": [],
                "prompt": [],
                self.config.task_name_field: [],
            }

            for i in range(len(example["input"])):
                input = example["input"][i]
                outputs = example["outputs"][i]
                type = example["type"][i]
                subject = example[self.config.task_name_field][i]

                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": AVAILABLE_PROMPTS[type],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                source = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": input + "\n\n" + AVAILABLE_PROMPTS[type],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                for output in outputs:
                    return_dict["source"].append(source)
                    return_dict["target"].append(output)
                    return_dict["prompt"].append(prompt)
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

        out = expand_targets_and_chat(train_dataset[:2])
        train_dataset = train_dataset.map(
            expand_targets_and_chat,
            batched=True,
            batch_size=1000,
            desc="Applying chat template...",
            remove_columns=["input", "outputs", "type"],
        )

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset


@dataclass
class PrefixDatasetConfig(KMDatasetConfig):
    # Given a passage, what proportion to use as a label ?
    label_frac: float = 0.23  # to match summary proportion
    # Once we have a label / target within a passage, how
    # many tokens to use as "warmup" ?
    prefix_length: int = 64


class PrefixDataCollator(DefaultCollator):
    def __call__(self, batch):

        assert "input_ids" in batch[0], "dataset should be tokenized"

        # TODO: Do we want to pack / pad samples differently with / without context ?
        return self.packed_collate(batch)


@DataModule.register("prefix_km", config_cls=PrefixDatasetConfig)
class PrefixDatasetModule(DataModule):
    collate_class = PrefixDataCollator
    # for this dataset, we will always pre-tokenize the inputs
    # so that we can split prefix / labels in tokens vs words

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        assert "train" in dataset

        def tokenize_datapoint(data):
            out = {}
            for key in data.keys():
                if key == "input":
                    toked = self.tokenizer(
                        data[key],
                        max_length=self.config.prefix_length,
                        padding="longest",
                        pad_to_multiple_of=1,
                        return_tensors="np",
                    )
                    out["input_ids"] = toked["input_ids"].squeeze().astype("int64")
                    out["attention_mask"] = (
                        toked["attention_mask"].squeeze().astype("float32")
                    )
                    out["seq_lens"] = np.array([len(out["input_ids"])])
                else:
                    out[key] = data[key]
            return out

        dataset = dataset.map(tokenize_datapoint, batched=False, num_proc=20)

        # finally, we build the label. First, we filter out any point that is shorter than
        # the desired prefix length
        dataset = dataset.filter(
            lambda x: len(x["input"]) >= self.config.prefix_length, num_proc=20
        )

        # next, we build `label_ids`
        def build_label_ids(data):
            all_tokens = data["input_ids"]
            label_ids = all_tokens[1:]
            input_ids = all_tokens[:-1]
            data["attention_mask"] = data["attention_mask"][:-1]

            # input ids will be split into
            # [ context ] [warmup] [labels]
            # where the distillation loss will be measured on the [labels]
            # the input to the frozen model will be [context] [warmup] [labels]
            # the input to the KM model will be [warmup] [labels]

            label_start = int(len(input_ids) * self.config.label_frac)
            warmup_start = label_start - self.config.prefix_length

            # ensure that nothing before `label_start` has loss computed
            label_ids[:label_start] = [-100] * label_start

            data["input_ids"] = input_ids
            data["nc_input_ids"] = input_ids[warmup_start:]
            data["label_ids"] = label_ids
            data["nc_label_ids"] = label_ids[warmup_start:]
            data["nc_attention_mask"] = data["attention_mask"][warmup_start:]

            return data

        dataset = dataset.map(build_label_ids, batched=False, num_proc=20)

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
        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset
