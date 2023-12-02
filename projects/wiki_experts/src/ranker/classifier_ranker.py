import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from projects.wiki_experts.src.ranker.experts_ranker import ExpertsRanker
from projects.wiki_experts.src.config import tasks_names_to_ids_ada, tasks_names_to_ids


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class T5Classifier(pl.LightningModule):
    def __init__(
        self,
    ):
        super().__init__()

        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def forward(self, inputs, labels=None):
        outputs = self.model(inputs, labels=labels)
        return outputs

    def training_step(self, batch):
        inputs = batch["input_ids"]
        labels = batch["labels"]

        outputs = self(inputs, labels=labels)
        loss = outputs.loss
        return loss


class SentenceTransformerClassifier(ExpertsRanker):
    # define the classifier, the x is the input, the task_id or expert_id is the label
    def __init__(
        self,
        num_labels,
        hidden_size=768,
        transformer_embed_dim=384,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_encoder = self.text_encoder_init(requires_grad=False)
        self.num_labels = num_labels
        # linear text encoder
        self.text_projecter = nn.Linear(transformer_embed_dim, hidden_size)
        self.out_projecter = nn.Linear(hidden_size, num_labels)
        if num_labels == 439:
            self.tasks_names_to_ids = tasks_names_to_ids_ada
        else:
            self.tasks_names_to_ids = tasks_names_to_ids
        self.save_hyperparameters(ignore=["text_encoder"])

    def text_encoder_init(self, requires_grad=False):
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

        # frozen the transformer parameters
        auto_model = text_encoder._first_module().auto_model
        for param in auto_model.parameters():
            param.requires_grad = requires_grad
        return text_encoder

    def forward(self, x):
        # Encode the text input
        text_output = torch.tensor(self.text_encoder.encode(x)).to(device)
        # conver the text output to hidden vector
        text_output_projecter = self.text_projecter(text_output)
        # Calculate the logits
        logits = self.out_projecter(text_output_projecter)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        text_input, task_name = batch["input"], batch["task_name"]
        label = torch.tensor([self.tasks_names_to_ids[task] for task in task_name]).to(
            device
        )
        logits = self(text_input)
        loss = F.cross_entropy(logits, label)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["input"]),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        text_input, task_name = batch["input"], batch["task_name"]
        label = torch.tensor([self.tasks_names_to_ids[task] for task in task_name]).to(
            device
        )
        logits = self(text_input)
        loss = F.cross_entropy(logits, label)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["input"]),
        )
        return loss

    def test_step(self, batch, batch_idx):
        text_input, task_name = batch["input"], batch["task_name"]
        label = torch.tensor([self.tasks_names_to_ids[task] for task in task_name]).to(
            device
        )
        logits = self(text_input)
        loss = F.cross_entropy(logits, label)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["input"]),
        )

        # compute the accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == label).item() / len(label)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)
        return loss


if __name__ == "__main__":
    from projects.wiki_experts.src.ranker.classification_module import (
        ClassificationDataModuleFlatMultiTask,
        ClassificationConfig,
    )

    from pytorch_lightning import Trainer

    from projects.wiki_experts.src.config import ExpertConfig
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os

    config = ExpertConfig.parse()
    dm = ClassificationDataModuleFlatMultiTask(
        ClassificationConfig(
            dataset=config.dataset,
            model=config.model,
            finetune_task_name=config.finetune_task_name,
            train_batch_size=config.train_batch_size,
            predict_batch_size=config.predict_batch_size,
        )
    )

    module = SentenceTransformerClassifier(num_labels=439)
    module.to(device)
    # for batch in dm.train_dataloader():
    #     loss = model.training_step(batch, 0)
    #     print(loss)
    #     break
    task_name = "adversarial_qa_dbert_answer_the_following_q"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd() + f"/checkpoints/{task_name}",
        save_top_k=1,
        verbose=True,
        monitor="val/loss",
        mode="min",
        filename=f"{task_name}" + "-{val/loss:.004f}",
        save_last=True,
    )

    # trainer = Trainer(
    #     callbacks=[checkpoint_callback],
    #     max_epochs=3,
    #     max_steps=20,
    #     # val_check_interval=10,
    # )
    # trainer.fit(module, dm)

    # model = module.from_pretrained(
    #     "/projects/futhark1/data/wzm289/code/lucas_mttl/projects/wiki_experts/checkpoints/adversarial_qa_dbert_answer_the_following_q/last.ckpt"
    # )
    breakpoint()
