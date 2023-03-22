from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from mttl.datamodule.ni_data_module import CollateWrapperFn
from mttl.dataloader.xfit_dataset_readers import XFitDatasetReader
from mttl.datamodule import IndexConcatDataset
from mttl.utils import get_example_to_ids, get_tasks_list


class XFitDataModule(LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.predict_batch_size,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=CollateWrapperFn(self.pad_token_id),
        )
        
    @property
    def all_instructions(self):
        raise NotImplementedError()

    def __init__(self, config):
        super().__init__()

        self.config = config

        if not self.config.finetune_task_name:
            self.tasks = get_tasks_list(config.custom_tasks_splits, "train")
            self.task2id = {task: idx for idx, task in enumerate(self.tasks)}
        else:
            self.task_name = self.config.finetune_task_name
            self.task2id = {self.task_name: 0}
            self.tasks = [self.task_name]
            assert (
                self.config.task_prefix
            ), "A prefix should be set if finetuning on a task for XFIT"
        self.id2task = {idx: task for task, idx in self.task2id.items()}

        if config.example_to_ids_path:
            self.example2id = get_example_to_ids(config.example_to_ids_path)
        else:
            self.example2id = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, model_max_length=config.max_input_length
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        if config.embeddings_path:
            task_embed_path = config.embeddings_path
        else:
            task_embed_path = None

        self.dataset_reader = XFitDatasetReader(
            config.train_dir,
            self.tokenizer,
            tasks=self.tasks,
            task_prefix=config.task_prefix,
            task2id=self.task2id,
            example2id=self.example2id,
            task_embed_path=task_embed_path,
            max_input_length=config.max_input_length,
            max_output_length=config.max_output_length,
            use_task_descriptions=config.use_task_descriptions,
        )

        print("Training on the following tasks: {}".format(self.tasks))

    def setup(self, stage=None):
        self.train_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("train")
        )
        self.val_dataset = IndexConcatDataset(
            self.dataset_reader.read_orig_datasets("dev")
        )
        
        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))
        
        if self.config.finetune_task_name:
            self.test_dataset = IndexConcatDataset(
                self.dataset_reader.read_orig_datasets("test")
            )
            print("Test steps:", len(self.test_dataloader()))


class XFitPretrainDataModule(XFitDataModule):
    pass


class XFitFinetuneDataModule(XFitDataModule):
    pass
