from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import get_args_parser
from callbacks import ProgressCallback
from data_module import NIPretrainDataModule, PretrainDataModule
from pl_model import EncoderDecoder


def run_multitask(args):
    seed_everything(args.seed, workers=True)

    # data
    if args.dataset == "xfit":
        dm = PretrainDataModule(args)
    elif args.dataset == "ni":
        dm = NIPretrainDataModule(args)
    else:
        raise NotImplementedError()

    # model + opt
    if args.n_tasks is None:
        args.n_tasks = len(dm.task2id)

    module = EncoderDecoder(**vars(args))

    if not args.finetune_full_model:
        for name, param in module.named_parameters():
            if param.requires_grad:
                print("Training parameter: ", name)

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=args.exp_name,
    )
    wandb_logger.experiment.save("*.py")

    # model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        filename="mbart-{epoch:02d}-{val/loss:.004f}",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        gpus=-1,
        accelerator="gpu",
        logger=wandb_logger,
        num_sanity_val_steps=0,
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=50,
        strategy="ddp_find_unused_parameters_false",
        limit_val_batches=1.0,
        callbacks=[ProgressCallback(), checkpoint_callback],
        accumulate_grad_batches=args.gradient_accumulation_steps,
        precision=int(args.precision)
        if args.precision in ["16", "32"]
        else args.precision,
    )
    trainer.fit(module, dm)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run_multitask(args)
