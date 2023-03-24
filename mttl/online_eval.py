import copy
import torch
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning import Trainer


class T0OnlineZeroShot(Callback):
    def __init__(self, every_steps):
        super().__init__()

        self.every_steps = every_steps

    def on_fit_start(self, trainer, pl_module) -> None:
        from mttl.datamodule.t0_data_module import T0FinetuneDataModule

        self.data = []
        for task in ["copa", "winogrande", "anli-r1"]:
            config = copy.deepcopy(pl_module.hparams)
            config.finetune_task_name = task

            self.data.append(T0FinetuneDataModule(config))
            self.data[-1].setup("fit")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        from mttl.models.t0_encoder_decoder import T0EncoderDecoder

        # create backup of the current pl_module weights
        # and restore backup at the end
        if trainer.global_step % self.every_steps == 0:            
            device = pl_module.device
            result = torch.zeros(len(self.data)).to(pl_module.device)

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

            del trainer

            pl_module.model = pl_module.model.to(device)
            pl_module.log(
                "test/zero_shot_perf",
                result.mean(),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
