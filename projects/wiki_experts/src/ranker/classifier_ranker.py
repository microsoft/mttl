import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

from mttl.models.utils import EfficientCheckpointModule
from projects.wiki_experts.src.ranker.adapter_ranker import AdapterRanker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentenceTransformerClassifier(AdapterRanker, EfficientCheckpointModule):
    # define the classifier, the x is the input, the task_id or expert_id is the label
    def __init__(
        self,
        task_names,
        hidden_size=768,
        transformer_embed_dim=384,
        temperature=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_encoder = self.text_encoder_init(requires_grad=False)
        self.ids_to_tasks_names = task_names
        self.task_names_to_ids = {task: i for i, task in enumerate(task_names)}
        self.num_labels = len(task_names)
        self.temperature = temperature
        # mask for available tasks
        self.available_mask: torch.Tensor = torch.ones(self.num_labels)

        # linear text encoder
        self.text_projecter = nn.Linear(transformer_embed_dim, hidden_size)
        self.out_projecter = nn.Linear(hidden_size, self.num_labels)
        self.save_hyperparameters(ignore=["text_encoder"])

    def forward(self, x):
        # Encode the text input
        text_output = torch.tensor(
            self.text_encoder.encode(x, show_progress_bar=False)
        ).to(device)
        # conver the text output to hidden vector
        text_output_projecter = self.text_projecter(text_output)
        # Calculate the logits
        logits = self.out_projecter(text_output_projecter)
        return logits

    def set_available_tasks(self, available_tasks):
        """Set the available tasks for the classifier."""
        self.available_mask.fill_(0.0)

        for task in available_tasks:
            if "default" in task:
                continue
            if (
                task in self.task_names_to_ids
            ):  # sometimes we train filtering classifiers on a subset of the tasks
                self.available_mask[self.task_names_to_ids[task]] = 1.0

    def predict_task(self, query, n=1):
        raise NotImplementedError("Not implemented yet.")

    @torch.no_grad()
    def predict_batch(self, batch, n=1):
        logits = self(batch["sources_texts"]).detach().cpu()

        if self.available_mask is not None:
            logits = logits + (1.0 - self.available_mask) * -100

        # safe softmax
        max_logits = torch.max(logits, dim=1, keepdim=True).values
        logits = logits - max_logits

        expert_indices = torch.topk(logits, k=n, dim=1)

        expert_prediction = [
            [self.ids_to_tasks_names[index.item()] for index in indices]
            for indices in expert_indices.indices
        ]
        expert_weights = [
            [weight.item() for weight in weights] for weights in expert_indices.values
        ]
        # increate the entropy of the weights
        expert_weights = np.array(expert_weights) / self.temperature
        expert_weights = np.exp(np.array(expert_weights))
        expert_weights = expert_weights / expert_weights.sum(axis=1, keepdims=True)

        return expert_prediction, expert_weights.tolist()

    def text_encoder_init(self, requires_grad=False, model_name="all-MiniLM-L6-v2"):
        text_encoder = SentenceTransformer(model_name)

        # frozen the transformer parameters
        auto_model = text_encoder._first_module().auto_model
        for param in auto_model.parameters():
            param.requires_grad = requires_grad
        return text_encoder

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        sources_texts, task_names = batch["sources_texts"], batch["task_names"]
        labels = torch.tensor([self.task_names_to_ids[task] for task in task_names]).to(
            device
        )
        logits = self(sources_texts)
        loss = F.cross_entropy(logits, labels)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        sources_texts, task_names = batch["sources_texts"], batch["task_names"]
        label = torch.tensor([self.task_names_to_ids[task] for task in task_names]).to(
            device
        )
        logits = self(sources_texts)
        loss = F.cross_entropy(logits, label)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )
        return loss

    def test_step(self, batch, batch_idx):
        sources_texts, task_names = batch["sources_texts"], batch["task_names"]
        label = torch.tensor([self.task_names_to_ids[task] for task in task_names]).to(
            device
        )
        logits = self(sources_texts)
        loss = F.cross_entropy(logits, label)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )

        # compute the accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == label).item() / len(label)
        self.log("test/acc", acc, on_epoch=True, prog_bar=True)
        return loss


class T5Classifier(SentenceTransformerClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def text_encoder_init(self, requires_grad=False, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)

        # Freeze all the parameters
        for param in model.parameters():
            param.requires_grad = requires_grad
        return model

    def forward(self, x):
        # Encode the text input
        input_ids = self.tokenizer(
            x, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).input_ids.to(device)

        last_hidden_states = self.text_encoder.encoder(
            input_ids=input_ids
        ).last_hidden_state

        # Pooling strategy: here, we just take the representation of the first token
        pooled_output = last_hidden_states[:, 0]

        # Calculate the logits
        text_output_projecter = self.text_projecter(pooled_output)
        # Calculate the logits
        logits = self.out_projecter(text_output_projecter)
        return logits
