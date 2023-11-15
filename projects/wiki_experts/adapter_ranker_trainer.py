import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from classification_module import ClassificationDataModule
from models.classifer_ranker import Classifier


def train_model():
    # using wandb project
    wandb_logger = pl.loggers.WandbLogger(project="wiki_experts")

    text_encoder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )  # You need to define your text encoder

    # frozen the transformer parameters
    auto_model = text_encoder._first_module().auto_model
    for param in auto_model.parameters():
        param.requires_grad = False

    # train the classifier
    datamodule = ClassificationDataModule(batch_size=1024)
    classifier = Classifier(text_encoder, num_labels=246)

    # add model checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss_epoch",
        dirpath="classification_ranker/",
        filename="classifier-{epoch:02d}-{val/loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        devices=1,
        logger=wandb_logger,
    )
    trainer.fit(classifier, datamodule)
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    train_model()
