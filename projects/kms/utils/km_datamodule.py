import math
import os
import re
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


def create_dcd_pairs(tokenizer, source, target, prompt, apply_chat_template=True):
    """Create DCD pairs, teacher sees the source + prompt, student sees only the prompt."""
    teacher_source = source + "\n\n" + prompt
    student_source = prompt

    if apply_chat_template:
        teacher_source = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": teacher_source,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        student_source = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": student_source,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return teacher_source, student_source


@dataclass
class KMDatasetConfig(DatasetConfig):
    # there might be multiple types, i.e. "qa", "summary", or maybe else in the future
    use_only_type: str = "summary"
    # for each input example, we could have several outputs (e.g. several summaries or QA pairs), we can only use N of these
    num_outputs_per_chunk: int = -1
    # field in the dataset that contains the task name
    task_name_field: str = "document_id"
    # field in the dataset that contains the task source
    task_source_field: str = "document_id"
    # for train / dev split, split by document chunk, or split the list of summary / q/a's ?
    split_train_dev_on: str = "document_chunk"
    # use chat template if available
    use_chat_template: bool = True


class KMDataCollator(DefaultCollator):
    def __call__(self, batch):
        if "input_ids" in batch[0]:
            return self.packed_collate(batch)

        output_batch = super().__call__(batch)

        nc_sources = [b["nc_source"] for b in batch]
        labels = [b["target"] for b in batch]
        nc_batch = self.prepare_inputs_for_gpt_family(nc_sources, labels)

        # no context tensors used for context distillation loss
        output_batch["nc_input_ids"] = nc_batch["input_ids"]
        output_batch["nc_attention_mask"] = nc_batch["attention_mask"]
        output_batch["nc_labels"] = nc_batch["labels"]
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
                    if not isinstance(output, dict):
                        # NOTE: some old dataformats have the output as a string
                        # this is a hack to convert it to a dict for `summary` and `qa`
                        assert isinstance(output, str)
                        if type_ == "summary":
                            output = {"summary": output}
                        elif type_ == "qa":
                            output = output.replace("**", "")
                            # regex to match on on the string 'Question:' or the string 'Answer:"
                            pattern = r"(?:Question:\s*(.*?)\s*Answer:\s*(.*))"
                            match = re.search(
                                pattern, output, re.DOTALL | re.IGNORECASE
                            )
                            try:
                                output = {
                                    "question": match.group(1).strip(),
                                    "answer": match.group(2).strip(),
                                }
                            except Exception as e:
                                logger.warning(f"Failed to parse QA pair: {output}")
                                continue
                        else:
                            raise ValueError(
                                f"Unknown type {type_} for output {output}"
                            )
                    if type(output) == dict:
                        if type_ == "qa":
                            prompt_str = "Generate a question-answer pair given the preceding passage."
                            output_str = f"\nQuestion: {output['question']}\nAnswer: {output['answer']}"
                        elif type_ == "q":
                            prompt_str = (
                                "Generate a question given the preceding passage."
                            )
                            output_str = f"{output['question']}"
                        elif type_ == "a":
                            prompt_str = f"Answer the following question given the preceding passage.\nQuestion: {output['question']}"
                            output_str = f"{output['answer']}"
                        elif type_ == "summary":
                            prompt_str = "Summarize the preceding passage."
                            output_str = output["summary"]
                        elif type_ == "entigraph":
                            prompt_str = "Given the preceding passage, discuss the relation between a pair of entities"
                            output_str = f"{output['entigraph']}"
                    else:
                        prompt_str = "Summarize the preceding passage."
                        output_str = output

                    context_source, no_context_source = create_dcd_pairs(
                        self.tokenizer,
                        input,
                        output_str,
                        prompt_str,
                        apply_chat_template=self.config.use_chat_template
                        and self.tokenizer.chat_template is not None,
                    )
                    return_dict["source"].append(context_source)
                    return_dict["nc_source"].append(no_context_source)
                    return_dict["target"].append(output_str)

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


class ChunkDataset(torch.utils.data.Dataset):
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

        if self.config.label_frac:
            label_start = max(1, int(len(input_ids) * (1 - self.config.label_frac)))
        else:
            label_start = max(1, len(input_ids) - self.config.label_n_tokens)
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
    label_n_tokens: int = None
    # Once we have a label / target within a passage, how
    # many tokens to use as "warmup" ?
    prefix_length: int = 0
    # field in the dataset that contains the task source
    text_field: str = "text"


class LMDataCollator(DefaultCollator):
    def __call__(self, batch):

        assert "input_ids" in batch[0], "dataset should be tokenized"

        return self.packed_collate(batch)


@DataModule.register("doc_km", config_cls=LMDatasetConfig)
class LMDataModule(DataModule):
    collate_class = LMDataCollator
    # for this dataset, we will always pre-tokenize the inputs
    # so that we can split prefix / labels in tokens vs words

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.config.label_frac is None or self.config.label_n_tokens is None
        assert (
            self.config.label_frac is not None or self.config.label_n_tokens is not None
        )

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
                if key == self.config.text_field:
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

        self.train_dataset = ChunkDataset(
            train_dataset, self.config, deterministic=False
        )
        self.dev_dataset = ChunkDataset(dev_dataset, self.config, deterministic=True)
        self.test_dataset = None


@dataclass
class ConcatDatasetConfig(KMDatasetConfig):
    max_concat_tokens: int = None
    use_only_type: str = "summary"


# Chunked Summary Dataset
@DataModule.register("concat_km", config_cls=ConcatDatasetConfig)
class ConcatDatasetModule(KMDatasetModule):

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

        def expand_targets_and_chat_cs(example):
            return_dict = defaultdict(list)
            # only keep 'summary' outputs; gather at most self.config.n_concat for nc_source
            allowed_types = self.config.use_only_type.split(",")

            for i in range(len(example["input"])):
                if example["type"][i] not in allowed_types:
                    continue

                input = example["input"][i]
                outputs = example["outputs"][i]
                concat_outputs = [
                    (
                        "".join(list(["" if v is None else v for v in o.values()]))
                        if isinstance(o, dict)
                        else o
                    )
                    for o in outputs
                ]
                len_in_tokens = [len(self.tokenizer.encode(o)) for o in concat_outputs]
                for s_idx in range(len(outputs)):
                    # summary at index i will appear first. Now, sample `n_concat - 1` other indices
                    other_idx = [j for j in range(len(outputs)) if j != s_idx]
                    # We will use the indices in `synthetic_idx` to create the synthetic data
                    synthetic_data = []
                    synthetic_idx = [s_idx] + other_idx
                    total_tokens = 0

                    # let's make sure the total number of concatenated tokens is less than `max_concat_tokens`, but
                    # that it at least includes 1 summary
                    upper_bound = max(
                        1,
                        (
                            np.cumsum(len_in_tokens) < self.config.max_concat_tokens
                        ).sum(),
                    )
                    for idx in synthetic_idx[:upper_bound]:

                        if example["type"][i] == "summary":
                            prompt_str = "Summarize the preceding passage."
                            synthetic_data.append(
                                outputs[idx]["summary"]
                                if isinstance(outputs[idx], dict)
                                else outputs[idx]
                            )
                        elif example["type"][i] == "qa":
                            if isinstance(outputs[idx], dict):
                                output_str = f"\nQuestion: {outputs[idx]['question']}\nAnswer: {outputs[idx]['answer']}"
                            elif isinstance(outputs[idx], str):
                                output_str = outputs[idx]
                            else:
                                raise TypeError("invalid type for output")
                            prompt_str = "Generate a question-answer pair given the preceding passage."
                            synthetic_data.append(output_str)
                        else:
                            raise ValueError(f"Unknown type {example['type'][i]}")

                        # Let's exit here so that at least one summary is always present
                        total_tokens += len_in_tokens[idx]
                        if total_tokens > self.config.max_concat_tokens:
                            break

                    join_str = "\n\n\n\n"
                    concat_data = join_str.join(synthetic_data)
                    context_source, no_context_source = create_dcd_pairs(
                        self.tokenizer,
                        input,
                        concat_data,
                        prompt_str,
                        apply_chat_template=self.config.use_chat_template
                        and self.tokenizer.chat_template is not None,
                    )
                    return_dict["source"].append(context_source)
                    return_dict["nc_source"].append(no_context_source)
                    return_dict["target"].append(concat_data)
                    return_dict[self.config.task_name_field].append(
                        example[self.config.task_name_field][i]
                    )
            return return_dict

        train_dataset = train_dataset.map(
            expand_targets_and_chat_cs,
            batched=True,
            batch_size=1000,
            desc="Applying concatenated summaries...",
            remove_columns=train_dataset.column_names,
        )

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )

        self.test_dataset = self.dev_dataset


@dataclass
class FullDocKMDatasetConfig(KMDatasetConfig):
    pass


@DataModule.register("full_doc_km", config_cls=FullDocKMDatasetConfig)
class FullDocKMDatasetModule(DataModule):
    collate_class = DefaultCollator

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        assert "train" in dataset

        def apply_doc_prompt(example):
            return_dict = defaultdict(list)

            # breakpoint()
            for i in range(len(example["text"])):
                doc = example["text"][i]
                prompt = f"You are given a partial document. You must predict the next token in the document. Start predicting as soon as the document starts.\n\nDocument : {doc}"

                # Apply chat template
                if (
                    self.config.use_chat_template
                    and self.tokenizer.chat_template is not None
                ):
                    formatted_text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    formatted_text = prompt

                return_dict["source"].append(formatted_text)
                return_dict["target"].append(doc)  # The target is the document itself
                return_dict[self.config.task_name_field].append(
                    example[self.config.task_name_field][i]
                )

            return return_dict

        # Filter dataset by task if needed
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

        # Apply chat template to each document
        train_dataset = train_dataset.map(
            apply_doc_prompt,
            batched=True,
            batch_size=1000,
            desc="Applying document prompt template...",
            remove_columns=train_dataset.column_names,
        )

        # Split into train/dev
        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset
