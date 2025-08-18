import os
from dataclasses import dataclass
from functools import partial

import numpy
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.models.library.dataset_library import DatasetLibrary
from datasets import load_dataset

def apply_template_phi3(tokenizer, example):
    message = [
        {"role": "user", "content": example["source"]},
        {"role": "assistant", "content": example["target"]},
    ]

    tokenized_message = tokenizer.apply_chat_template(message, tokenize=False)
    # split the message into source and target
    example["source"] = tokenized_message.split("<|assistant|>")[0]
    example["target"] = "<|assistant|>" + tokenized_message.split("<|assistant|>")[1]

    return example


def apply_template_mistral(tokenizer, example):
    message = [
        {"role": "user", "content": example["source"]},
        {"role": "assistant", "content": example["target"]},
    ]
    tokenized_message = tokenizer.apply_chat_template(message, tokenize=False)
    example["source"] = tokenized_message.split("[/INST]")[0]
    example["target"] = "[/INST]" + tokenized_message.split("[/INST]")[1]
    return example
def apply_template(dataset, tokenizer, model_name):
    if "Phi-3-mini-4k-instruct" in model_name:
        dataset = dataset.map(
            partial(apply_template_phi3, tokenizer),
        num_proc=int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16)),
        )
    elif "Mistral-7B-Instruct-v0.3" in model_name:
        dataset = dataset.map(
            partial(apply_template_mistral, tokenizer),
        num_proc=int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16)),
        )
    return dataset


@dataclass
class TaskAdapterConfig(DatasetConfig):
    source_template: str = None
    dataset: str = "zhan1993/task_adapter_dataset"


@DataModule.register("task_adapter", config_cls=TaskAdapterConfig)
class TaskAdapterModule(DataModule):
    def setup_dataset(self):
        self.dataset = load_dataset(self.config.dataset)

        # filter out the examples with source length == 0
        self.dataset = self.dataset.filter(lambda x: len(x["source"]) > 0)

        n_proc = min(
            len(self.dataset), int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))
        )
        self.dataset = apply_template(self.dataset, self.tokenizer, self.config.model)

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            test_dataset,
        ) = maybe_filter_hf_dataset_by_task(
            self.dataset, "task_name", self.config.finetune_task_name, n_proc=n_proc
        )

        self.train_dataset = train_dataset
        if test_dataset is not None and len(test_dataset) > 0:
            self.dev_dataset = test_dataset
            self.test_dataset = test_dataset
        else:
            self.dev_dataset = train_dataset
            self.test_dataset = train_dataset


if __name__ == "__main__":
    from transformers import AutoTokenizer
    config = TaskAdapterConfig(
        dataset="zhan1993/task_adapter_dataset",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        train_batch_size=1,
        finetune_task_name="mmlu",
    )
    data_module = TaskAdapterModule(config)

    # conversation = [{"role": "user", "content": "What's the weather like in Paris?"}, {"role": "assistant", "content": "The weather in Paris is sunny."}]
    # tokenizer = AutoTokenizer.from_pretrained(config.model)
    # tokenized_message = tokenizer.apply_chat_template(conversation, tokenize=False)
    # print(tokenized_message)
    # breakpoint()
    data_module.setup_dataset()
    count = 0
    for batch in tqdm(data_module.test_dataloader()):
        print(batch)
        breakpoint()
