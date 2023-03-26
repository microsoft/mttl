import copy
import torch
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning import Trainer


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
        if batch_idx % self.every_steps == 0:
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

            for i, online_data in enumerate(self.data):
                results = trainer.test(ft_wrapper, datamodule=online_data)[0]
                result[i] = results["test/acc_0shot"]

            for i, task in enumerate(self.EARLY_STOP_TASKS):
                es_result[i] = result[self.TASKS.index(task)]

            del trainer

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
