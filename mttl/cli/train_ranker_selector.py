import os

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from mttl.callbacks import LiveCheckpointCallback
from mttl.config import RankerConfig
from mttl.datamodule.clip_data_module import (
    CLIPExpertsConfig,
    CLIPExpertsDatamodule,
    CLIPTripleDataModule,
)
from mttl.datamodule.mt_seq_to_seq_module import FlanConfig, FlanModule
from mttl.models.ranker.classifier_ranker import (
    ClassifierSmooth,
    SentenceTransformerClassifier,
)
from mttl.models.ranker.clip_ranker import CLIPRanker, CLIPTripletRanker


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
        encoder_model_name=args.model,
        text_embedding_dim=args.text_embedding_dim,
        expert_embedding_dim=args.expert_embedding_dim,
        projection_dim=args.projection_dim,
        learning_rate=args.learning_rate,
    )
    if args.ranker_path:
        model = model.load_from_checkpoint(args.ranker_path)

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=f"{args.output_dir}/classification_ranker_{args.exp_name}/",
        monitor="val/loss_epoch",
        save_last=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
        val_check_interval=args.val_check_interval,
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
        encoder_model_name=args.model,
        text_embedding_dim=args.text_embedding_dim,
    )

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=f"{args.output_dir}/classification_ranker_{args.exp_name}/",
        monitor="val/loss_epoch",
        save_last=True,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
        val_check_interval=args.eval_every,
    )

    trainer.fit(model, datamodule)
    if wandb_logger:
        wandb_logger.experiment.finish()


def train_classifier(args: RankerConfig):
    seed_everything(args.seed, workers=True)

    wandb_logger = None
    if os.environ.get("WANDB_API_KEY") or args.wandb_project:
        import wandb

        project = os.environ.get("WANDB_PROJECT", "wiki_experts")
        project = args.wandb_project if args.wandb_project is not None else project
        args.exp_name = "dev_run" if args.exp_name is None else args.exp_name

        wandb_logger = pl.loggers.WandbLogger(
            project=project,
            name=args.exp_name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb_logger.experiment.save("*.py")
        wandb_logger.experiment.save("*/*.py")

    # train the classifier
    if "flan" not in args.dataset_type:
        raise ValueError("Only FLAN supported for now.")

    datamodule = FlanModule(args.dataset_config)

    if args.ranker_path:
        module = SentenceTransformerClassifier.from_pretrained(args.ranker_path)

    module = SentenceTransformerClassifier(
        task_names=datamodule.task_names,
        encoder_model_name=args.model,
        transformer_embed_dim=args.text_embedding_dim,
    )

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=f"{args.output_dir}/classification_ranker_{args.exp_name}/",
        monitor="val/loss_epoch",
        save_last=True,
        mode="min",
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.num_train_epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
    )
    trainer.fit(module, datamodule)
    trainer.test(module, datamodule.test_dataloader())

    if wandb_logger:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    from mttl.config import RankerConfig

    args = RankerConfig.parse()

    if args.ranker_model == "classifier":
        train_classifier(args)
    elif args.ranker_model == "clip":
        train_clip(args)
    elif args.ranker_model == "clip_triplet":
        train_triplet_clip(args)
