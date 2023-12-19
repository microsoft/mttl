import pytorch_lightning as pl
from projects.wiki_experts.src.ranker.classifier_ranker import (
    SentenceTransformerClassifier,
    T5Classifier,
)
from mttl.datamodule.mt_seq_to_seq_module import (
    FlatMultiTaskConfig,
    FlatMultiTaskModule,
)
from projects.wiki_experts.src.ranker.clip_ranker import CLIPRanker, CLIPTripletRanker
from projects.wiki_experts.src.ranker.clip_data_module import (
    CLIPExpertsDatamodule,
    CLIPExpertsConfig,
    CLIPTripleDataModule,
)
import os
from pytorch_lightning import seed_everything


def train_triplet_clip(args):
    seed_everything(args.seed, workers=True)
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
        wandb_logger.experiment.save("*/ranker/*.py")
    # test the model
    dataconfig = CLIPExpertsConfig(
        dataset=args.dataset,
        model=args.model,
        train_batch_size=args.train_batch_size,
        finetune_task_name=args.finetune_task_name,
        predict_batch_size=args.predict_batch_size,
    )
    datamodule = CLIPTripleDataModule(dataconfig)
    task_names = datamodule.task_names

    if "default" not in task_names:
        task_names.append("default")
    model = CLIPTripletRanker(
        task_names=task_names,
        encoder_model_name=args.encoder_model_name,
        text_embedding_dim=args.text_embedding_dim,
    )

    # add model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # monitor="val/loss_epoch",
        dirpath=f"clip_triplet_ranker_{args.exp_name}/",
        filename="clip-{epoch:02d}-{val/loss:.2f}",
        save_top_k=1,
        # mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
        val_check_interval=0.5,
    )
    trainer.fit(model, datamodule)
    if wandb_logger:
        wandb_logger.experiment.finish()


def train_clip(args):
    seed_everything(args.seed, workers=True)
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

    # test the model
    dataconfig = CLIPExpertsConfig(
        dataset=args.dataset,
        model=args.model,
        train_batch_size=args.train_batch_size,
        finetune_task_name=args.finetune_task_name,
        predict_batch_size=args.predict_batch_size,
    )

    datamodule = CLIPTripleDataModule(dataconfig)
    task_names = datamodule.task_names

    if "default" not in task_names:
        task_names.append("default")

    model = CLIPRanker(
        task_names=task_names,
        encoder_model_name=args.encoder_model_name,
        text_embedding_dim=args.text_embedding_dim,
    )
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
        val_check_interval=0.5,
    )
    trainer.fit(model, datamodule)
    if wandb_logger:
        wandb_logger.experiment.finish()


def train_classifier(args):
    # using wandb project
    seed_everything(args.seed, workers=True)
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
        predict_batch_size=args.predict_batch_size,
    )
    datamodule = FlatMultiTaskModule(config)
    if args.ranker_path:
        if args.ranker_model == "classifier":
            module = SentenceTransformerClassifier.from_pretrained(args.ranker_path)
        elif args.ranker_model == "t5":
            module = T5Classifier.from_pretrained(args.ranker_path)
        else:
            raise ValueError("Only classifier and t5 models supported for now.")
    else:
        if args.ranker_model == "classifier":
            module = SentenceTransformerClassifier(task_names=datamodule.task_names)
        elif args.ranker_model == "t5":
            module = T5Classifier(
                task_names=datamodule.task_names, transformer_embed_dim=512
            )
        else:
            raise ValueError("Only classifier and t5 models supported for now.")

    # add model checkpoint

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        dirpath=f"classification_ranker_{args.exp_name}",
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
        val_check_interval=0.5,
        limit_val_batches=0.5,
    )
    # trainer.fit(module, datamodule)
    trainer.test(module, datamodule.test_dataloader())
    if wandb_logger:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    from projects.wiki_experts.src.ranker.config import RankerConfig

    args = RankerConfig.parse()
    if args.ranker_model == "classifier" or args.ranker_model == "t5":
        train_classifier(args)
    elif args.ranker_model == "clip":
        train_clip(args)
    elif args.ranker_model == "clip_triplet":
        train_triplet_clip(args)
