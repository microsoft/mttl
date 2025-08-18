from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.base import DefaultCollator
from dataclasses import dataclass
import os
from mttl.models.library.dataset_library import DatasetLibrary
from functools import partial
import torch
@dataclass
class AbstentionDataConfig(DatasetConfig):
    predict_output_dir: str = "./"

@dataclass

class DataCollatorForAbstention(DefaultCollator):
    def __call__(self, batch):
        if "input_ids" in batch[0]:
            return self.packed_collate(batch)

        # Otherwise process as expected
        sources = [b["source"] for b in batch]
        labels = [b["target"] for b in batch]
        task_ids = [b.get(self.task_id_field, None) for b in batch]
        task_names = [b.get(self.task_name_field, None) for b in batch]
        task_sources = [b.get(self.task_source_field, None) for b in batch]
        ids = [b['id'] for b in batch]
        output_batch = (
            self.prepare_inputs_for_gpt_family(sources, labels)
            if self.model_family == "gpt"
            else self.prepare_inputs_for_seq2seq_family(sources, labels)
        )

        has_task_sources = all(ts is not None for ts in task_sources)
        has_task_names = all(tn is not None for tn in task_names)
        has_task_ids = all(tid is not None for tid in task_ids)

        if not has_task_ids and has_task_names and self.task_to_id:
            output_batch["task_ids"] = torch.LongTensor(
                [self.task_to_id[tn] for tn in task_names]
            )
        elif has_task_ids:
            output_batch["task_ids"] = torch.LongTensor(task_ids)

        if has_task_names and not has_task_sources:
            task_sources = task_names

        output_batch["task_names"] = task_names
        output_batch["sources_texts"] = sources
        output_batch["labels_texts"] = labels
        output_batch["task_sources"] = task_sources
        # output_batch['ids'] = ids

        # integrate extra fields in the batch if required
        if self.collate_extra_fields:
            for field in self.collate_extra_fields:
                output_batch[field] = [b[field] for b in batch]

        return output_batch

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

DataModule.register("abstention", config_cls=AbstentionDataConfig)
class AbstentionDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 4))
        dataset = DatasetLibrary.pull_dataset(self.config.dataset)
        dataset = dataset.rename_column("question", "source")
        dataset = dataset.rename_column("answer", "target")

        dataset = apply_template(dataset, self.tokenizer, self.config.model)
        self.train_dataset = dataset["train"]

        self.dev_dataset = self.test_dataset = dataset["train"]

    @property
    def collate_fn(self):
        return DataCollatorForAbstention(
            tokenizer=self.tokenizer,
            padding="longest",
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


if __name__ == "__main__":
    config = AbstentionDataConfig(
        model="microsoft/Phi-3-mini-4k-instruct",
        dataset="zhan1993/coconot_original_train_routing",
    )

    datamodule = AbstentionDataModule(config, for_generation=True)

    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
