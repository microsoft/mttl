import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from classification_module import ClassificationDataModule
from models.classifer_ranker import Classifier
from projects.wiki_experts.src.config import ExpertConfig
import os


def train_model(args):
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

    text_encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # You need to define your text encoder

    # frozen the transformer parameters
    auto_model = text_encoder._first_module().auto_model
    for param in auto_model.parameters():
        param.requires_grad = False

    # train the classifier
    datamodule = ClassificationDataModule(batch_size=args.train_batch_size)
    classifier = Classifier(text_encoder, num_labels=args.num_labels)

    # add model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        dirpath="classification_ranker/",
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
    trainer.fit(classifier, datamodule)
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    args = ExpertConfig.parse()
    train_model(args)
