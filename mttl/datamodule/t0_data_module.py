from typing import Dict
import torch
import numpy as np
import random
from pytorch_lightning import LightningDataModule
from mttl.datamodule.base import DefaultCollator
from mttl.datamodule.utils import get_tokenizer

from mttl.utils import template_to_string
from mttl.datamodule import IndexConcatDataset
from mttl.dataloader.t0_dataset_readers import get_dataset_reader


def apply_template(template, example, hash_friendly=False, handle_edge_cases=True):
    """Could be done with a context manager, but we're lazy."""
    if hash_friendly:
        state = random.getstate()
        random.seed(42)

    input_target_str = template.apply(example)

    if handle_edge_cases:
        if len(input_target_str) == 2:
            input_str, target_str = input_target_str
            if target_str == "":
                target_str = "<NO LABEL>"
        else:
            input_str = "<NO INPUT>"
            target_str = "<NO LABEL>"
    else:
        input_str, target_str = input_target_str

    if hash_friendly:
        random.setstate(state)

    return input_str, target_str


class T0FinetuneDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tokenizer = get_tokenizer(config)

        self.collate_fn = MultiChoiceCollator(
            tokenizer=self.tokenizer,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family="seq2seq",
        )

        self.dataset_reader = get_dataset_reader(config)
        self.task_to_id = {config.finetune_task_name: 0}

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        _ = self.dataset_reader.read_few_shot_dataset()

    @property
    def all_instructions(self):
        """Return all task instructions used in the dataset."""
        instruction_list = []
        for template in self.dataset_reader.get_train_template():
            instruction_list.append(template_to_string(template))
        for template in self.dataset_reader.get_eval_template():
            instruction_list.append(template_to_string(template))
        return instruction_list

    @property
    def full_dataset(self):
        assert self.train_dataset

        all_datasets_with_template = []
        for template in self.dataset_reader.get_train_template():
            all_datasets_with_template.append(
                T0FinetuneDatasetWithTemplate(
                    self.train_dataset,
                    template,
                    self.tokenizer,
                    self.config.finetune_task_name,
                )
            )
        for template in self.dataset_reader.get_eval_template():
            all_datasets_with_template.append(
                T0FinetuneDatasetWithTemplate(
                    self.dev_dataset,
                    template,
                    self.tokenizer,
                    self.config.finetune_task_name,
                )
            )
        return IndexConcatDataset(all_datasets_with_template)

    def setup(self, stage):
        self.train_dataset = self.dataset_reader.read_few_shot_dataset()
        self.dev_dataset = self.dataset_reader.read_orig_dataset("validation")

        # randomly sample templates
        self.train_dataset_wt = T0FinetuneDatasetWithTemplate(
            self.train_dataset,
            self.dataset_reader.get_train_template(),
            self.tokenizer,
            self.config.finetune_task_name,
        )
        self.dev_dataset_wt = T0FinetuneDatasetWithTemplate(
            self.dev_dataset,
            self.dataset_reader.get_eval_template(),
            self.tokenizer,
            self.config.finetune_task_name,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset_wt,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=min([self.config.train_batch_size, 8]),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset_wt,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=min([self.config.predict_batch_size, 8]),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class T0PretrainDataModule(LightningDataModule):
    def __init__(self, config, flatten_templates=False):
        super().__init__()

        self.config = config
        self.tokenizer = get_tokenizer(config)

        self.collate_fn = DefaultCollator(
            tokenizer=self.tokenizer,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
            model_family="seq2seq",
        )

        self.dataset_reader = get_dataset_reader(config)
        self.base_templates = self.dataset_reader.get_template()
        self.base_tasks = self.dataset_reader.get_base_tasks()
        self.task_to_id = {t: i for i, t in enumerate(self.base_tasks)}
        self.id_to_task = dict((k, v) for v, k in self.task_to_id.items())
        self.flatten_templates = flatten_templates

    @property
    def full_dataset(self):
        return self._full_dataset

    @property
    def all_instructions(self):
        """Return all task instructions used in the dataset."""
        instruction_list = []
        for template in self.base_templates:
            instruction_list.append(template_to_string(template))
        return instruction_list

    def setup(self, stage="fit"):
        self.train_datasets = self.dataset_reader.read_orig_dataset("train")
        self.train_datasets_withtemplate = []
        self.val_datasets_withtemplate = []
        self.all_datasets_withtemplate = []
        self.rng = torch.Generator().manual_seed(self.config.seed)

        for train_dataset, base_template in zip(
            self.train_datasets, self.base_templates
        ):
            if self.dataset_reader.config.use_t0_templates_as_tasks:
                task_name = f"{train_dataset.dataset_name}/{train_dataset.subset_name}/{train_dataset.template_name}"
                task_id = self.task_to_id[task_name]
            else:
                task_name = f"{train_dataset.dataset_name}/{train_dataset.subset_name}"
                task_id = self.task_to_id[task_name]

            tr_dataset_wt = T0PretrainDatasetWithTemplate(
                train_dataset,
                base_template,
                self.tokenizer,
                task_id,
                task_name,
            )

            # take 100 examples for validation
            tr_dataset_wt, val_dataset_wt = torch.utils.data.random_split(
                tr_dataset_wt,
                [
                    len(tr_dataset_wt) - 100,
                    100,
                ],
                generator=self.rng,
            )

            self.train_datasets_withtemplate.append(tr_dataset_wt)
            self.val_datasets_withtemplate.append(val_dataset_wt)
            self.all_datasets_withtemplate.extend([tr_dataset_wt, val_dataset_wt])

        if self.config.use_t0_few_shot_training_set:
            for i in range(len(self.train_datasets_withtemplate)):
                # pick only 100 examples for training per template x task
                self.train_datasets_withtemplate[i] = torch.utils.data.random_split(
                    self.train_datasets_withtemplate[i],
                    [
                        len(self.train_datasets_withtemplate[i]) - 1000,
                        1000,
                    ],
                )[1]

        self._full_dataset = IndexConcatDataset(self.all_datasets_withtemplate)
        self.train_dataset = IndexConcatDataset(self.train_datasets_withtemplate)
        self.val_dataset = IndexConcatDataset(self.val_datasets_withtemplate)

        # store relevant information
        self.dataset_ids = [
            f"{x.dataset_name}_{x.subset_name}" for x in self.train_datasets
        ]
        self.template_names = [x.template_name for x in self.train_datasets]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=min([self.config.train_batch_size, 8]),
            persistent_workers=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=min([self.config.predict_batch_size, 8]),
            persistent_workers=True,
        )


class T0PretrainDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates, tokenizer, ds_id, task_name):
        super().__init__()

        self.ds_id = ds_id
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer
        self.task_name = task_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, example_id):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates

        example = self.dataset[example_id]
        source, target = apply_template(template, example, hash_friendly=False)

        return {
            "source": source,
            "target": target,
            "example_id": example_id,
            "task_id": self.ds_id,
            "task_name": self.task_name,
        }


class T0FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        dataset,
        templates,
        tokenizer,
        add_special_tokens=True,
        ds_id=0,
        task_name=None,
    ):
        super().__init__()

        self.ds_id = ds_id
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, example_id):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[example_id]

        source, target = apply_template(
            template, example, hash_friendly=False, handle_edge_cases=False
        )
        answer_choices = template.get_answer_choices_list(example)
        label = example["label"]
        idx = example["idx"]

        return {
            "source": source,
            "target": target,
            "answer_choices": answer_choices,
            "label": label,
            "idx": idx,
            "example_id": example_id,
            "task_id": self.ds_id,
            "task_name": self.task_name,
        }


class MultiChoiceCollator(DefaultCollator):
    """
    wrapper of `collate_fn` in `create_collate_fn` that is ddp compatible
    """

    def __call__(self, batch: Dict):
        output_batch = super().__call__(batch)

        answer_choices = [b["answer_choices"] for b in batch]
        flat_answer_choice = [
            choice for list_choices in answer_choices for choice in list_choices
        ]
        answer_choices_ = self.tokenizer(
            flat_answer_choice,
            max_length=self.max_output_length,
            padding=self.padding,
            return_tensors=self.return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        num_choice = [len(list_choices) for list_choices in answer_choices]
        if max(num_choice) != min(num_choice):
            raise NotImplementedError(
                "The collate_fn is not implmented for variable number of choices"
            )
        answer_choices_ids = (
            answer_choices_["input_ids"]
            .view(len(answer_choices), max(num_choice), -1)
            .contiguous()
        )

        output_batch["idx"] = torch.LongTensor([b["idx"] for b in batch])
        output_batch["labels"] = torch.LongTensor([b.label for b in batch])
        output_batch["answer_choices_ids"] = answer_choices_ids
        return output_batch
