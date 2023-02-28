import copy
import torch
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning import Trainer


class OnlineZeroShot(Callback):
    def __init__(self, every_steps):
        super().__init__()

        self.every_steps = every_steps

    def on_fit_start(self, trainer, pl_module) -> None:
        from datamodule.ni_data_module import NIDataModule

        params = copy.deepcopy(pl_module.hparams)
        params.custom_tasks_splits = "./dataloader/ni_data/test_tasks_20.txt"

        self.online_data = NIDataModule(params)
        self.online_data.setup()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0) -> None:
        from models.encoder_decoder import Finetuner

        # create backup of the current pl_module weights
        # and restore backup at the end
        if trainer.global_step > 0 and trainer.global_step % self.every_steps == 0:
            device = pl_module.device
            result = torch.zeros(1).to(pl_module.device)

            ft_wrapper = Finetuner(**pl_module.hparams, tokenizer=pl_module.tokenizer, model_object=pl_module.model)
            trainer = Trainer(
                gpus=-1,
                accelerator="gpu",
                num_sanity_val_steps=0,
                enable_checkpointing=False,
            )
            results = trainer.test(ft_wrapper, datamodule=self.online_data)[0]
            result[0] = results["test/metric_perf"]
            del trainer

            pl_module.model = pl_module.model.to(device)
            pl_module.log(
                "test/zero_shot_perf",
                result[0],
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
