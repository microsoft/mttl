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
                    else:
                        # fallback to default prompt
                        prompt_str = "Summarize the preceding passage."
                        output_str = output

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


class DocumentDataset(torch.utils.data.Dataset):
    """Dataset to handle randomly chunking documents at every epoch"""

    def preprend_a_zero(self, tensor):
        return torch.cat([torch.zeros_like(tensor[:1]), tensor])

    def __init__(self, docs, config, deterministic=False):
        self.docs = docs
        self.config = config
        self.deterministic = deterministic

        from mttl.dist_utils import ddp_rank, ddp_world_size

        self.rank = ddp_rank
        self.world_size = ddp_world_size
        chunk_size = self.config.max_input_length

        if deterministic:
            # we always want to pick the first index of every chunk
            self.chunks_per_doc = torch.LongTensor(
                [max(1, len(doc["input_ids"]) // chunk_size) for doc in self.docs]
            )
            self.cum_chunks_per_doc = torch.cumsum(self.chunks_per_doc, 0).long()

            # self.cum_chunks_per_doc[i] = number of chunks from documents 0 to i-1
            self.cum_chunks_per_doc = self.preprend_a_zero(self.cum_chunks_per_doc)
        else:
            # Every document can be indexed at multiple places
            self.max_start_idx = torch.LongTensor(
                [len(doc["input_ids"]) - chunk_size for doc in self.docs]
            )
            self.cum_start_idx = torch.cumsum(self.max_start_idx, 0).long()

            # self.cum_start_idx[i] = sum of tokens from documents 0 to i-1
            self.cum_start_idx = self.preprend_a_zero(self.cum_start_idx)

    def build_labels(self, datapoint):
        all_tokens = datapoint["input_ids"]

        # clone the data
        input_ids = all_tokens[:]
        label_ids = all_tokens[:]

        # datapoint["attention_mask"] = datapoint["attention_mask"][:-1]

        # input ids will be split into
        # [ context ] [warmup] [labels]
        # where the distillation loss will be measured on the [labels]
        # the input to the frozen model will be [context] [warmup] [labels]
        # the input to the KM model will be [warmup] [labels]

        label_start = max(0, int(len(input_ids) * (1 - self.config.label_frac)))
        warmup_start = max(0, label_start - self.config.prefix_length)

        # ensure that nothing before `label_start` has loss computed
        label_ids[:label_start] = [-100] * label_start

        datapoint["labels"] = label_ids
        datapoint["nc_input_ids"] = input_ids[warmup_start:]
        datapoint["nc_labels"] = label_ids[warmup_start:]
        datapoint["nc_attention_mask"] = datapoint["attention_mask"][warmup_start:]

        return datapoint

    def __getitem__(self, idx):
        """Enable both random sampling and sequential iteration over document chunks"""

        if self.deterministic:
            doc_idx = torch.where(
                (idx >= self.cum_chunks_per_doc[:-1])
                & (idx < self.cum_chunks_per_doc[1:])
            )[0].item()
            # undo the offset from the previous documents
            chunk_idx = (idx - self.cum_chunks_per_doc[doc_idx]).item()
            start_idx = chunk_idx * self.config.max_input_length
        else:
            doc_idx = torch.where(
                (idx >= self.cum_start_idx[:-1]) & (idx < self.cum_start_idx[1:])
            )[0].item()
            # undo the offset from the previous documents
            start_idx = (idx - self.cum_start_idx[doc_idx]).item()

        worker_id = getattr(get_worker_info(), "id", -1)
        end_idx = start_idx + self.config.max_input_length

        # The following assert will be executed for a document with a single chunk smaller than max_input_length
        assert end_idx < len(self.docs[doc_idx]["input_ids"])
        print(
            f"rank {self.rank}, worker id {worker_id}, det {self.deterministic}",
            start_idx,
        )

        output = {
            "input_ids": self.docs[doc_idx]["input_ids"][start_idx:end_idx],
            "labels": self.docs[doc_idx]["input_ids"][start_idx:end_idx],
            "attention_mask": self.docs[doc_idx]["attention_mask"][start_idx:end_idx],
            "seq_lens": [self.config.max_input_length],
            "task_names": self.docs[doc_idx]["document_id"],
        }

        return self.build_labels(output)

    def __len__(self):
        if not hasattr(self, "_len"):
            if self.deterministic:
                # always pick the first index of every chunk
                self._len = sum(self.chunks_per_doc)
            else:
                # How many potential starting indices are there ?
                self._len = (self.max_start_idx + 1).sum().item()

        return self._len


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

        self.train_dataset = DocumentDataset(
            train_dataset, self.config, deterministic=False
        )
        self.dev_dataset = DocumentDataset(dev_dataset, self.config, deterministic=True)
        self.test_dataset = None
