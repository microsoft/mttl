import torch
import numpy as np
import random
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer

from mttl.utils import hash_example
from mttl.datamodule import IndexConcatDataset
from mttl.dataloader.t0_dataset_readers import get_dataset_reader
from mttl.dataloader.data_utils import ExampleInfo, MultiChoiceExampleInfo


def apply_template(template, example, hash_friendly=False, handle_edge_cases=True):
    """Could be done with a context manager, but we're lazy.
    """
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

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.tokenizer.model_max_length = config.max_input_length

        self.dataset_reader = get_dataset_reader(config)
        self.task2id = {config.finetune_task_name: 0}

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        _ = self.dataset_reader.read_few_shot_dataset()

    def flattened_datasets(self):
        assert self.train_dataset

        all_datasets_with_template = []
        for template in self.dataset_reader.get_train_template():
            all_datasets_with_template.append(
                T0FinetuneDatasetWithTemplate(
                    self.train_dataset,
                    template,
                    self.tokenizer,
                )
            )
        for template in self.dataset_reader.get_eval_template():
            all_datasets_with_template.append(
                T0FinetuneDatasetWithTemplate(
                    self.dev_dataset,
                    template,
                    self.tokenizer,
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
        )
        self.dev_dataset_wt = T0FinetuneDatasetWithTemplate(
            self.dev_dataset,
            self.dataset_reader.get_eval_template(),
            self.tokenizer,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset_wt,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            drop_last=True,
            num_workers=min([self.config.train_batch_size, 8]),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset_wt,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.predict_batch_size, 8]),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset_wt,
            batch_size=self.config.predict_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.predict_batch_size, 8]),
            persistent_workers=True,
        )


class T0FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(
        self,
        dataset,
        templates,
        tokenizer,
        add_special_tokens=True,
        ds_id=0,
    ):
        super().__init__()

        self.ds_id = ds_id
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, example_id):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[example_id]

        input_str, target_str = apply_template(template, example, hash_friendly=False, handle_edge_cases=False)
        answer_choices = template.get_answer_choices_list(example)

        if isinstance(input_str, list):
            input_ids = torch.cat(
                [
                    self.tokenizer(
                        input_field,
                        return_tensors="pt",
                        truncation=True,
                        add_special_tokens=False,
                    ).input_ids.squeeze(0)
                    for input_field in input_str[:-1]
                ]
                + [
                    self.tokenizer(
                        input_str[-1],
                        return_tensors="pt",
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                    ).input_ids.squeeze(0)
                ]
            )
        else:
            input_ids = self.tokenizer(
                input_str,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=self.add_special_tokens,
            ).input_ids.squeeze(0)

        target_ids = self.tokenizer(
            target_str,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        ).input_ids.squeeze(0)

        answer_choices_ids = [
            self.tokenizer(
                answer_choice,
                return_tensors="pt",
                truncation=True,
                add_special_tokens=self.add_special_tokens,
            ).input_ids.squeeze(0)
            for answer_choice in answer_choices
        ]

        label = torch.LongTensor([example["label"]])
        idx = torch.LongTensor([example["idx"]])

        input_str, _ = apply_template(template, example, hash_friendly=True, handle_edge_cases=False)

        hash = hash_example(
            " ".join(input_str) if isinstance(input_str, list) else input_str
        )
        return MultiChoiceExampleInfo(
            input_ids,
            target_ids,
            answer_choices_ids,
            label,
            idx,
            self.ds_id,
            hash,
            example_id,
        )


class T0PretrainDataModule(LightningDataModule):
    def __init__(self, config, flatten_templates=False):
        super().__init__()

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.tokenizer.model_max_length = config.max_input_length

        self.dataset_reader = get_dataset_reader(config)
        self.base_templates = self.dataset_reader.get_template()
        self.base_tasks = self.dataset_reader.get_base_tasks()
        self.task2id = {t: i for i, t in enumerate(self.base_tasks)}
        self.id2task = dict((k, v) for v, k in self.task2id.items())
        self.flatten_templates = flatten_templates

    def setup(self, stage='fit'):
        self.train_datasets = self.dataset_reader.read_orig_dataset("train")
        self.train_datasets_withtemplate = []
        self.val_datasets_withtemplate = []
        self.all_datasets_withtemplate = []
        self.rng = torch.Generator().manual_seed(self.config.seed)

        for train_dataset, base_template in zip(
            self.train_datasets, self.base_templates
        ):
            if self.dataset_reader.config.use_t0_templates_as_tasks:
                task_id = self.task2id[
                    f"{train_dataset.dataset_name}/{train_dataset.subset_name}/{train_dataset.template_name}"
                ]
            else:
                task_id = self.task2id[
                    f"{train_dataset.dataset_name}/{train_dataset.subset_name}"
                ]

            tr_dataset_wt = T0PretrainDatasetWithTemplate(
                train_dataset,
                base_template,
                self.tokenizer,
                task_id,
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
            self.train_datasets_withtemplate.append(
                tr_dataset_wt
            )
            self.val_datasets_withtemplate.append(
                val_dataset_wt
            )
            self.all_datasets_withtemplate.extend([tr_dataset_wt, val_dataset_wt])

        if self.config.use_t0_few_shot_training_set:
            for i in range(len(self.train_datasets_withtemplate)):
                # pick only 100 examples for training per template x task
                self.train_datasets_withtemplate[i] = torch.utils.data.random_split(
                    self.train_datasets_withtemplate[i],
                    [
                        len(self.train_datasets_withtemplate[i]) - 1000,
                        1000,
                    ]
                )[1]

        self.full_train_dataset = IndexConcatDataset(self.all_datasets_withtemplate)
        self.train_dataset = IndexConcatDataset(
            self.train_datasets_withtemplate
        )
        self.val_dataset = IndexConcatDataset(
            self.val_datasets_withtemplate
        )

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
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=True),
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
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=True),
            drop_last=True,
            num_workers=min([self.config.predict_batch_size, 8]),
            persistent_workers=True,
        )


class T0PretrainDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates, tokenizer, ds_id):
        super().__init__()

        self.ds_id = ds_id
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, example_id):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates

        example = self.dataset[example_id]

        input_str, target_str = apply_template(template, example, hash_friendly=False)
        input_ids = self.tokenizer(
            input_str, return_tensors="pt", truncation=True
        ).input_ids.squeeze(0)
        target_ids = self.tokenizer(
            target_str, return_tensors="pt", truncation=True
        ).input_ids.squeeze(0)

        # we need to be hash friendly here, template.apply is non-deterministic :-(
        hash = hash_example(apply_template(template, example, hash_friendly=True)[0])
        return ExampleInfo(
            input_ids, target_ids, self.ds_id, hash, example_id, input_text=input_str
        )


class CollateMultiChoiceFnWrapper:
    """
    wrapper of `collate_fn` in `create_collate_fn` that is ddp compatible
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: MultiChoiceExampleInfo):
        pad_token_id = self.pad_token_id

        input_ids = [b.input_ids for b in batch]
        target_ids = [b.target_ids for b in batch]
        task_ids = [b.task_id for b in batch]
        hashes = [b.hash for b in batch]
        answer_choices_ids = [b.answer_choices_ids for b in batch]
        labels = [b.label for b in batch]
        idx = [b.idx for b in batch]

        task_ids = torch.LongTensor(task_ids)
        labels = torch.cat(labels)
        idx = torch.cat(idx)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        target_ids = torch.nn.utils.rnn.pad_sequence(
            target_ids, batch_first=True, padding_value=pad_token_id
        )
        flat_answer_choice_ids = [
            choice for list_choices in answer_choices_ids for choice in list_choices
        ]
        num_choice = [len(list_choices) for list_choices in answer_choices_ids]
        if max(num_choice) != min(num_choice):
            raise NotImplementedError(
                "The collate_fn is not implmented for variable number of choices"
            )
        flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
            flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
        )
        answer_choices_ids = flat_answer_choices_ids.view(
            len(answer_choices_ids), max(num_choice), -1
        ).contiguous()

        output_batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "task_ids": task_ids,
            "answer_choices_ids": answer_choices_ids,
            "labels": labels,
            "idx": idx,
            "hashes": hashes,
        }
        return output_batch


class CollatePretrainFnWrapper:
    """
    wrapper of `collate_fn` in `create_collate_fn` that is ddp compatible
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        from utils import trim_batch

        pad_token_id = self.pad_token_id
        input_ids = [b.input_ids for b in batch]
        target_ids = [b.target_ids for b in batch]
        task_ids = [b.task_id for b in batch]
        ex_ids = [b.example_id for b in batch]
        hashes = [b.hash for b in batch]

        task_ids = torch.tensor(task_ids, dtype=torch.long)
        ex_ids = torch.tensor(ex_ids, dtype=torch.long)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        target_ids = torch.nn.utils.rnn.pad_sequence(
            target_ids, batch_first=True, padding_value=pad_token_id
        )
        # trim batch, just in case
        input_ids = trim_batch(input_ids, self.pad_token_id)
        target_ids = trim_batch(target_ids, self.pad_token_id)

        output_batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "task_ids": task_ids,
            "ex_ids": ex_ids,
            "hashes": hashes,
        }
        return output_batch


def create_collate_fn(pad_token_id, pretrain):
    if pretrain:
        return CollatePretrainFnWrapper(pad_token_id)
    return CollateMultiChoiceFnWrapper(pad_token_id)
