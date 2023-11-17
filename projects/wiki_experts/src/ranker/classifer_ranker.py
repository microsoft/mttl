import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifer(pl.LightningModule):
    # define the classifier, the x is the input, the task_id or expert_id is the label
    def __init__(
        self, text_encoder, num_labels, hidden_size=768, transformer_embed_dim=384
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.num_labels = num_labels
        # linear text encoder
        self.text_projecter = nn.Linear(transformer_embed_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.save_hyperparameters(ignore=["text_encoder"])

    def forward(self, x):
        # Encode the text input
        text_output = torch.tensor(self.text_encoder.encode(x)).to(device)
        # conver the text output to a 768-dimensional vector
        text_output_projecter = self.text_projecter(text_output)
        # Calculate the logits
        logits = self.classifier(text_output_projecter)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        text_input, label = batch["input"], batch["label"]
        logits = self(text_input)
        loss = F.cross_entropy(logits, label)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text_input, label = batch["input"], batch["label"]
        logits = self(text_input)
        loss = F.cross_entropy(logits, label)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        text_input, label = batch["input"], batch["label"]
        logits = self(text_input)
        loss = F.cross_entropy(logits, label)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # compute the accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == label).item() / len(label)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)
        return loss
