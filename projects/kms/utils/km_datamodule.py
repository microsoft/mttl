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
from mttl.dist_utils import ddp_rank, ddp_world_size
from mttl.logging import logger
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class KMDatasetConfig(DatasetConfig):
    # there might be multiple types, i.e. "qa", "summary", or "next_chunk" or maybe else in the future
    use_only_type: str = "summary"
    # for each input example, we could have several outputs (e.g. several summaries or QA pairs), we can only use N of these
    num_outputs_per_chunk: int = -1
    # field in the dataset that contains the task name
    task_name_field: str = "document_id"
    # field in the dataset that contains the task source
    task_source_field: str = "document_id"
    # for train / dev split, split by document chunk, or split the list of summary / q/a's ?
    split_train_dev_on: str = "document_chunk"
    # flip inputs and outputs
    flip_inputs_outputs: bool = False


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
            allowed_types = self.config.use_only_type.split(",")

            for i in range(len(example["input"])):
                input = example["input"][i]
                outputs = example["outputs"][i]
                type_ = example["type"][i]
                subject = example[self.config.task_name_field][i]

                if type_ not in allowed_types:
                    if type_ == "qa" and "q" in allowed_types:
                        type_ = "q"
                    elif type_ == "qa" and "a" in allowed_types:
                        type_ = "a"
                    else:
                        continue

                for i, output in enumerate(outputs):
                    # for QA, we want to show the question as well as the prompt
                    if type(output) == dict:
                        if type_ == "qa":
                            if self.config.flip_inputs_outputs:
                                prompt_str = "Generate a passage containing the preceding question-answer pair."
                            else:
                                prompt_str = "Generate a question-answer pair given the preceding passage."
                            output_str = f"\nQuestion: {output['question']}\nAnswer: {output['answer']}"
                        elif type_ == "q":
                            if self.config.flip_inputs_outputs:
                                prompt_str = "Generate a passage containing the preceding question."
                            else:
                                prompt_str = (
                                    "Generate a question given the preceding passage."
                                )
                            output_str = f"{output['question']}"
                        elif type_ == "a":
                            if self.config.flip_inputs_outputs:
                                prompt_str = "Generate a passage containing the preceding answer."
                            else:
                                prompt_str = f"Answer the following question given the preceding passage.\nQuestion: {output['question']}"
                            output_str = f"{output['answer']}"
                        elif type_ == "summary":
                            if self.config.flip_inputs_outputs:
                                prompt_str = "Generate a passage which can be summarized by the previous summary."
                            else:
                                prompt_str = "Summarize the preceding passage."
                            output_str = output["summary"]
                        elif type_ == "next_chunk":
                            if self.config.flip_inputs_outputs:
                                raise ValueError(
                                    "Cannot flip inputs and outputs for `next_chunk`"
                                )
                            else:
                                prompt_str = "Generate the continuation of the previous passage, make sure to keep the same style and narrative flow."
                            output_str = output["next_chunk"]
                    else:
                        # fallback to default prompt
                        if type_ == "qa":
                            if self.config.flip_inputs_outputs:
                                prompt_str = "Generate a passage containing the preceding question-answer pair."
                            else:
                                prompt_str = "Generate a question-answer pair given the preceding passage."
                        elif type_ == "summary":
                            if self.config.flip_inputs_outputs:
                                prompt_str = "Generate a passage which can be summarized as follows."
                            else:
                                prompt_str = "Summarize the preceding passage."
                        elif type_ == "next_chunk":
                            if self.config.flip_inputs_outputs:
                                raise ValueError(
                                    "Cannot flip inputs and outputs for `next_chunk`"
                                )
                            else:
                                prompt_str = "Generate the continuation of the previous passage, make sure to keep the same style and narrative flow."
                        else:
                            raise ValueError(
                                f"For legacy dataset, only qa, summary and next_chunk are supported!"
                            )
                        output_str = output

                    if self.config.flip_inputs_outputs:
                        input, output_str = output_str, input

                    prompt_str = self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": prompt_str,
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    source_str = self.tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": input + "\n\n" + prompt_str,
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    return_dict["source"].append(source_str)
                    return_dict["target"].append(output_str)
                    return_dict["prompt"].append(prompt_str)

                    if self.config.split_train_dev_on == "output":
                        # ensure that 95% or at least 1 point goes to dev
                        if i < max(int(len(outputs) * 0.05), 1):
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


class DocumentDatasetTwo(torch.utils.data.Dataset):
    def _build_structure(self):
        self.chunk_document = []
        self.chunk_positions = []
        self.chunk_lengths = []

        for doc_idx, doc_data in enumerate(self.docs):
            chunk = 0
            doc = doc_data["input_ids"]

            if not self.deterministic:
                while chunk < max(1, len(doc) - self.chunk_length):
                    self.chunk_document.append(doc_idx)
                    self.chunk_positions.append(chunk)
                    length = self.chunk_length
                    if chunk + self.chunk_length > len(doc):
                        length = len(doc) - chunk
                    self.chunk_lengths.append(length)
                    chunk += 1
            else:
                start_position = 0
                while chunk < math.ceil(len(doc) / self.chunk_length):
                    self.chunk_document.append(doc_idx)
                    self.chunk_positions.append(start_position)
                    length = min(len(doc) - start_position, self.chunk_length)
                    self.chunk_lengths.append(length)

                    if length < self.chunk_length:
                        start_position = max(0, len(doc) - self.chunk_length)
                    else:
                        start_position = start_position + self.chunk_length
                    chunk += 1

        self.total_positions = len(self.chunk_document)

    def __init__(self, documents, config, deterministic=False):
        self.docs = documents
        self.config = config
        self.chunk_length = config.max_input_length
        self.deterministic = deterministic
        self._build_structure()

    def build_labels(self, datapoint):
        all_tokens = datapoint["input_ids"]
        input_ids = all_tokens[:]
        label_ids = all_tokens[:]

        label_start = max(0, int(len(input_ids) * (1 - self.config.label_frac)))
        warmup_start = max(0, label_start - self.config.prefix_length)

        # ensure that nothing before `label_start` has loss computed
        label_ids[:label_start] = [-100] * label_start

        datapoint["labels"] = label_ids
        datapoint["nc_input_ids"] = input_ids[warmup_start:]
        datapoint["nc_labels"] = label_ids[warmup_start:]
        datapoint["nc_attention_mask"] = datapoint["attention_mask"][warmup_start:]
        return datapoint

    def __len__(self):
        return self.total_positions

    def __getitem__(self, idx):
        doc_idx = self.chunk_document[idx]
        start_idx = self.chunk_positions[idx]
        end_idx = start_idx + self.chunk_lengths[idx]
        output = {
            "input_ids": self.docs[doc_idx]["input_ids"][start_idx:end_idx],
            "labels": self.docs[doc_idx]["input_ids"][start_idx:end_idx],
            "attention_mask": self.docs[doc_idx]["attention_mask"][start_idx:end_idx],
            "seq_lens": [end_idx - start_idx],
            "task_names": self.docs[doc_idx]["document_id"],
        }
        return self.build_labels(output)


@dataclass
class LMDatasetConfig(KMDatasetConfig):
    # Given a passage, what proportion to use as a label ?
    label_frac: float = 0.23  # to match summary proportion
    # Once we have a label / target within a passage, how
    # many tokens to use as "warmup" ?
    prefix_length: int = 0


class LMDataCollator(DefaultCollator):
    def __call__(self, batch):

        assert "input_ids" in batch[0], "dataset should be tokenized"

        # TODO: Do we want to pack / pad samples differently with / without context ?
        return self.packed_collate(batch)


@DataModule.register("doc_km", config_cls=LMDatasetConfig)
class LMDataModule(DataModule):
    collate_class = LMDataCollator
    # for this dataset, we will always pre-tokenize the inputs
    # so that we can split prefix / labels in tokens vs words

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        assert "train" in dataset
        assert len(dataset) == 1, "all dataset should be in `train`"

        # Let's first filter out unused tasks
        (
            self._task_names,
            self._task_to_id,
            dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset,
            self.config.task_name_field,
            self.config.finetune_task_name,
            n_proc=n_proc,
        )

        def tokenize_datapoint(data):
            out = {}
            for key in data.keys():
                if key == "text":
                    toked = self.tokenizer(
                        data[key],
                        max_length=self.config.max_input_length,
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

        # For KMs, it doesn't make sense to split pool of docs into train and test
        # So let's split each doc into train and test
        def split_datapoint(data, split="train"):
            out = {}
            ex_len = len(data["input_ids"])
            split_idx = int(ex_len * 0.95)
            if split == "train":
                start_idx, end_idx = 0, split_idx
            else:
                start_idx, end_idx = split_idx, ex_len
            for key in data.keys():
                if key in ["input_ids", "attention_mask"]:
                    out[key] = data[key][start_idx:end_idx]
                else:
                    out[key] = data[key]
            return out

        train_dataset = dataset.map(
            partial(split_datapoint, split="train"), num_proc=20
        )
        dev_dataset = dataset.map(partial(split_datapoint, split="dev"), num_proc=20)

        self.train_dataset = DocumentDatasetTwo(
            train_dataset, self.config, deterministic=False
        )
        self.dev_dataset = DocumentDatasetTwo(
            dev_dataset, self.config, deterministic=True
        )
        self.test_dataset = None
