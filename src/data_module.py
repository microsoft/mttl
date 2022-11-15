import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataloader.fewshot_gym_multitask import NLPFewshotGymMultiTaskData
from dataloader.fewshot_gym_singletask import NLPFewshotGymSingleTaskData
from dataloader.ni_multitask_data import NIMultitaskData
from utils import get_ni_tasks_from_file, get_tasks_list, trim_batch


class PretrainDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        tasks = sorted(get_tasks_list(args.custom_tasks_splits, "all"))
        self.task2id = {task: idx for idx, task in enumerate(tasks)}
        self.train_tasks = get_tasks_list(args.custom_tasks_splits, "train")
        self.is_pretrain = torch.zeros(size=(len(self.task2id),)).bool()
        for task in self.train_tasks:
            self.is_pretrain[self.task2id[task]] = True
        self.training_on = self.is_pretrain.clone()

    def setup(self, stage=None):
        args = self.args
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.pad_token_id = tokenizer.pad_token_id
        print("Training on the following tasks: {}".format(self.train_tasks))

        if args.use_precomputed_task_embeddings:
            model = args.model.split("/")[-1]
            task_embed_path = f"task_embeds/xfit/{model}_embeds.pkl"
        else:
            task_embed_path = None

        if stage == "fit":
            train_ds = NLPFewshotGymMultiTaskData(
                self.args,
                self.args.train_dir,
                tasks=self.train_tasks,
                data_split="train",
                data_type="train",
                is_training=True,
                task2id=self.task2id,
                task_embed_path=task_embed_path,
                use_task_descriptions=args.use_task_descriptions,
            )
            val_ds = NLPFewshotGymMultiTaskData(
                self.args,
                self.args.train_dir,
                tasks=self.train_tasks,
                data_split="dev",
                data_type="dev",
                is_training=True,  # setting to true to that we can teacher force and get loss
                task2id=self.task2id,
                task_embed_path=task_embed_path,
                use_task_descriptions=args.use_task_descriptions,
            )

            train_ds.load_dataset(tokenizer)
            val_ds.load_dataset(tokenizer)

            self.train_wrapper = train_ds
            self.train_ds = train_ds.dataset

            self.val_wrapper = val_ds
            self.val_ds = val_ds.dataset
        else:
            raise ValueError

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.train_batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.predict_batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


class FinetuneDataModule(PretrainDataModule):
    def __init__(self, args):
        LightningDataModule.__init__(self)

        self.args = args
        self.task_name = self.args.finetune_task_name
        self.task2id = {self.task_name: 0}
        self.tasks = [self.task_name]
        self.is_pretrain = torch.zeros(size=(len(self.task2id),)).bool()
        for task in self.tasks:
            self.is_pretrain[self.task2id[task]] = True
        self.training_on = self.is_pretrain.clone()

    def setup(self, stage=None):
        args = self.args

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.pad_token_id = tokenizer.pad_token_id

        # get the tasks on which to train
        self.training_on.fill_(0)
        print("Training on the following tasks: {}".format(self.args.task_name))
        self.training_on[self.task2id[self.args.task_name]] = True

        if args.use_precomputed_task_embeddings:
            model = args.model.split("/")[-1]
            task_embed_path = f"task_embeds/xfit/{model}_embeds.pkl"
        else:
            task_embed_path = None

        if stage == "fit":
            train_ds = NLPFewshotGymSingleTaskData(
                self.args,
                self.args.task_name,
                self.args.train_file,
                data_type="train",
                is_training=True,
                task2id=self.task2id,
                task_embed_path=task_embed_path,
                use_task_descriptions=args.use_task_descriptions,
            )
            val_ds = NLPFewshotGymSingleTaskData(
                self.args,
                self.args.task_name,
                self.args.dev_file,
                data_type="dev",
                is_training=False,
                task2id=self.task2id,
                task_embed_path=task_embed_path,
                use_task_descriptions=args.use_task_descriptions,
            )

            train_ds.load_dataset(tokenizer)
            val_ds.load_dataset(tokenizer)

            self.train_wrapper = train_ds
            self.val_wrapper = val_ds

            self.train_ds = train_ds.dataset
            self.val_ds = val_ds.dataset

        elif stage == "test":
            test_ds = NLPFewshotGymSingleTaskData(
                self.args,
                self.args.task_name,
                self.args.test_file,
                data_type="test",  # TODO: check og code (used dev) [not used ?]
                is_training=False,
                task2id=self.task2id,
                task_embed_path=task_embed_path,
                use_task_descriptions=args.use_task_descriptions,
            )

            test_ds.load_dataset(tokenizer)
            self.test_wrapper = test_ds
            self.test_ds = test_ds.dataset

        else:
            raise ValueError

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.predict_batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )


class NIPretrainDataModule(LightningDataModule):
    def collate(self, batch):
        """Collate and trim to max length to speed-up inference.
        """
        result = torch.utils.data._utils.collate.default_collate(batch)
        result["input_ids"], result["attn_mask"] = trim_batch(
            result["input_ids"], self.pad_token_id, result["attn_mask"]
        )
        result["dec_input_ids"], result["dec_attn_mask"] = trim_batch(
            result["dec_input_ids"], self.pad_token_id, result["dec_attn_mask"]
        )
        result["defs_input_ids"], result["defs_attn_mask"] = trim_batch(
            result["defs_input_ids"], self.pad_token_id, result["defs_attn_mask"]
        )
        return result

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.args.train_batch_size,
            num_workers=16,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.args.predict_batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.collate,
        )

    def prepare_data(self):
        args = self.args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, model_max_length=args.max_input_length
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        if args.use_precomputed_task_embeddings:
            model = args.model.split("/")[-1]
            task_embed_path = f"task_embeds/ni/{model}_embeds.pkl"
        else:
            task_embed_path = None

        train_ds = NIMultitaskData(
            self.args.train_dir,
            tasks=self.tasks,
            data_split="train",
            is_training=True,
            task2id=self.task2id,
            task_embed_path=task_embed_path,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            use_task_descriptions=args.use_task_descriptions,
            num_positive_examples=args.num_pos_examples,
        )
        val_ds = NIMultitaskData(
            self.args.train_dir,
            tasks=self.tasks,
            data_split="dev",
            task2id=self.task2id,
            is_training=True,  # setting to true to that we can teacher force and get loss
            task_embed_path=task_embed_path,
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            use_task_descriptions=args.use_task_descriptions,
            num_positive_examples=args.num_pos_examples,
        )

        # this dumps the datasets to the cache directory
        train_ds.load_dataset(self.tokenizer)
        val_ds.load_dataset(self.tokenizer)

        self.train_ds = train_ds.dataset
        self.val_ds = val_ds.dataset
        self.train_wrapper = train_ds
        self.val_wrapper = val_ds

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tasks, self.task2id = get_ni_tasks_from_file(args.custom_tasks_splits)
        self.is_pretrain = torch.zeros(size=(len(self.task2id),)).bool()

        for task in self.tasks:
            self.is_pretrain[self.task2id[task]] = True

        print("Training on the following tasks: {}".format(self.tasks))
        self.training_on = self.is_pretrain.clone()
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        if self.train_ds is None:
            self.prepare_data()

        print("Training steps:", len(self.train_dataloader()))
        print("Validation steps:", len(self.val_dataloader()))


class NIFinetuneDataModule(NIPretrainDataModule):
    def __init__(self, args):
        LightningDataModule.__init__(self)

        self.args = args
        self.task_name = self.args.finetune_task_name
        self.task2id = {self.task_name: 0}
        self.tasks = [self.task_name]
        self.is_pretrain = torch.zeros(size=(len(self.task2id),)).bool()
        for task in self.tasks:
            self.is_pretrain[self.task2id[task]] = True
        self.training_on = self.is_pretrain.clone()

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.args.predict_batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate,
        )

    def prepare_data(self):
        super().prepare_data()

        test_ds = NIMultitaskData(
            self.args.train_dir,
            tasks=self.tasks,
            data_split="test",
            task2id=self.task2id,
            is_training=True,  # setting to true to that we can teacher force and get loss
            max_input_length=self.args.max_input_length,
            max_output_length=self.args.max_output_length,
            use_task_descriptions=self.args.use_task_descriptions,
            num_positive_examples=self.args.num_pos_examples,
        )
        test_ds.load_dataset(self.tokenizer)

        self.test_ds = test_ds.dataset
        self.test_wrapper = test_ds
