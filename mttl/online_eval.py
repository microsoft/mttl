import copy
import json

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from mttl.datamodule.ni_data_module import NiDataModule
from mttl.models.encoder_decoder import Finetuner


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

        self.val_data = NiDataModule(config, for_generation=True)

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
                model_object=pl_module.model,
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
            with open(
                pl_module.hparams.output_dir + f"/val_split_tasks_scores.jsonl", "a+"
            ) as f:
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
