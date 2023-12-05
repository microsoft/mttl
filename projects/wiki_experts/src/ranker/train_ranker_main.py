import pytorch_lightning as pl
from projects.wiki_experts.src.ranker.classifier_ranker import (
    SentenceTransformerClassifier,
)
from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)
from projects.wiki_experts.src.config import ExpertConfig
from projects.wiki_experts.src.ranker.clip_ranker import CLIPRanker, CLIPTripletRanker
from projects.wiki_experts.src.ranker.clip_data_module import (
    CLIPExpertsDatamodule,
    CLIPExpertsConfig,
    CLIPTripleDataModule,
)
import os


def train_triplet_clip(args):
    wandb_logger = None
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        import wandb

        project = os.environ.get("WANDB_PROJECT", "wiki_experts")
        project = args.wandb_project if args.wandb_project is not None else project
        args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=args.exp_name,  # , config=args_
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.experiment.save("*.py")
        wandb_logger.experiment.save("*/*.py")
    model = CLIPTripletRanker(expert_num=440)

    # test the model
    dataconfig = CLIPExpertsConfig(
        dataset=args.dataset,
        model=args.model,
        train_batch_size=args.train_batch_size,
        finetune_task_name=args.finetune_task_name,
        predict_batch_size=args.predict_batch_size,
    )
    datamodule = CLIPTripleDataModule(dataconfig)

    # add model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        dirpath=f"clip_ranker_{args.exp_name}/",
        filename="clip-{epoch:02d}-{val/loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
        val_check_interval=0.25,
    )
    trainer.fit(model, datamodule)
    if wandb_logger:
        wandb_logger.experiment.finish()


def train_clip(args):
    wandb_logger = None
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        import wandb

        project = os.environ.get("WANDB_PROJECT", "wiki_experts")
        project = args.wandb_project if args.wandb_project is not None else project
        args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=args.exp_name,  # , config=args_
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.experiment.save("*.py")
        wandb_logger.experiment.save("*/*.py")
    model = CLIPRanker()

    # test the model
    dataconfig = CLIPExpertsConfig(
        dataset=args.dataset,
        model=args.model,
        train_batch_size=args.train_batch_size,
        finetune_task_name=args.finetune_task_name,
        predict_batch_size=args.predict_batch_size,
    )
    datamodule = CLIPExpertsDatamodule(dataconfig)

    # add model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        dirpath=f"clip_ranker_{args.exp_name}/",
        filename="clip-{epoch:02d}-{val/loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
        val_check_interval=0.25,
    )
    trainer.fit(model, datamodule)
    if wandb_logger:
        wandb_logger.experiment.finish()


def train_classifier(args):
    # using wandb project
    wandb_logger = None
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        import wandb

        project = os.environ.get("WANDB_PROJECT", "wiki_experts")
        project = args.wandb_project if args.wandb_project is not None else project
        args.exp_name = "dev_run" if args.exp_name is None else args.exp_name
        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=args.exp_name,  # , config=args_
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.experiment.save("*.py")
        wandb_logger.experiment.save("*/*.py")

    # train the classifier
    if "flat" not in args.dataset:
        raise ValueError("Only flat datamodule supported for now.")

    config = FlatMultiTaskConfig(
        dataset=args.dataset,
        model=args.model,
        train_batch_size=args.train_batch_size,
        finetune_task_name=args.finetune_task_name,
    )
    datamodule = FlatMultiTaskModule(config)
    module = SentenceTransformerClassifier(
        labels=datamodule.task_names, text_encoder_trained=args.text_encoder_trained
    )

    # add model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        dirpath=f"classification_ranker_{args.dataset}",
        filename="classifier-{epoch:02d}-{val/loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
    )
    trainer.fit(module, datamodule)
    if wandb_logger:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    args = ExpertConfig.parse()
    if args.retrieval_model == "classifier":
        train_classifier(args)
    elif args.retrieval_model == "clip":
        train_clip(args)
    elif args.retrieval_model == "clip_triplet":
        train_triplet_clip(args)
