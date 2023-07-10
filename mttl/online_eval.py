import copy
import torch

from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer

from mttl.datamodule.ni_data_module import NIDataModule
from mttl.models.encoder_decoder import Finetuner
import json


class NIOnlineZeroShot(Callback):
    # evaluate 0-shot performance on a subset of test tasks, for early stopping
    VAL_TASKS = "./mttl/dataloader/ni_data/test_tasks_20.txt"

    # full 0-shot performance on test tasks
    TEST_TASKS = "./mttl/dataloader/ni_data/test_tasks.txt"

    def __init__(self, every_steps):
        super().__init__()

        self.every_steps = every_steps

    def on_fit_start(self, trainer, pl_module) -> None:
        config = copy.deepcopy(pl_module.hparams)
        config.custom_tasks_splits = self.VAL_TASKS
        config.predict_batch_size = 8

        self.val_data = NIDataModule(config)
        self.val_data.setup("fit")

        config = copy.deepcopy(pl_module.hparams)
        config.custom_tasks_splits = self.TEST_TASKS

        self.test_data = NIDataModule(config)
        self.test_data.setup("fit")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # create backup of the current pl_module weights
        # and restore backup at the end
        if batch_idx % (self.every_steps - 10) == 0 and batch_idx > 0:
            device = pl_module.device

            val_result = torch.zeros(1).to(pl_module.device)
            test_result = torch.zeros(1).to(pl_module.device)

            ft_wrapper = Finetuner(
                **pl_module.hparams,
                tokenizer=pl_module.tokenizer,
                model_object=pl_module.model
            )

            trainer = Trainer(
                gpus=-1,
                accelerator="gpu",
                num_sanity_val_steps=0,
                enable_checkpointing=False,
            )

            val_metrics = trainer.test(ft_wrapper, datamodule=self.val_data)[0]
            val_result[0] = val_metrics["test/metric_perf"]
            del trainer

            result_str = json.dumps(val_metrics) + "\n"
            with open(pl_module.hparams.output_dir + f"/val_split_tasks_scores.jsonl", "a+") as f:
                f.write(result_str)

            pl_module.model = pl_module.model.to(device)
            pl_module.log(
                "val/zero_shot_perf",
                val_result.mean(),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )


class T0OnlineZeroShot(Callback):
    TASKS = [
        "copa",
        "h-swag",
        "storycloze",
        "winogrande",
        "wsc",
        "wic",
        "rte",
        "cb",
        "anli-r1",
        "anli-r2",
        "anli-r3",
    ]
    
    EARLY_STOP_TASKS = [
        "copa",
        "winogrande",
        "anli-r1"
    ]

    def __init__(self, every_steps):
        super().__init__()

        self.every_steps = every_steps

    def on_fit_start(self, trainer, pl_module) -> None:
        from mttl.datamodule.t0_data_module import T0FinetuneDataModule

        self.data = []
        for task in self.TASKS:
            config = copy.deepcopy(pl_module.hparams)
            config.finetune_task_name = task

            self.data.append(T0FinetuneDataModule(config))
            self.data[-1].setup("fit")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        from mttl.models.t0_encoder_decoder import T0EncoderDecoder

        # create backup of the current pl_module weights
        # and restore backup at the end
        if batch_idx % 25_000 == 0 and batch_idx > 0:
            device = pl_module.device

            result = torch.zeros(len(self.data)).to(pl_module.device)
            es_result = torch.zeros(len(self.EARLY_STOP_TASKS)).to(pl_module.device)

            ft_wrapper = T0EncoderDecoder(
                **pl_module.hparams,
                tokenizer=pl_module.tokenizer,
                model_object=pl_module.model
            )
            trainer = Trainer(
                gpus=-1,
                accelerator="gpu",
                num_sanity_val_steps=0,
                enable_checkpointing=False,
            )

            all_results = []
            for i, online_data in enumerate(self.data):
                results = trainer.test(ft_wrapper, datamodule=online_data)[0]
                results["task_name"] = self.TASKS[i]
                results["step"] = batch_idx
                result[i] = results["test/acc_0shot"]
                all_results.append(results)

            for i, task in enumerate(self.EARLY_STOP_TASKS):
                es_result[i] = result[self.TASKS.index(task)]

            del trainer

            result_str = json.dumps(all_results) + "\n"
            with open(pl_module.hparams.output_dir + f"/online_zero_shot_scores.jsonl", "a+") as f:
                f.write(result_str)

            pl_module.model = pl_module.model.to(device)
            pl_module.log(
                "test/es_zero_shot_perf",
                es_result.mean(),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            pl_module.log(
                "test/zero_shot_perf",
                result.mean(),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
