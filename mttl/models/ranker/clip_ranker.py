# implements the CLIPRanker class
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from mttl.models.ranker.adapter_ranker import AdapterRanker
from mttl.models.utils import EfficientCheckpointModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    def __init__(
        self,
        trainable: bool = False,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()
        if model_name == "all-MiniLM-L6-v2":
            self.transformer_encoder = SentenceTransformer(
                model_name
            )  # You need to define your text encoder

            # frozen the transformer parameters
            auto_model = self.transformer_encoder._first_module().auto_model
            if not trainable:
                for param in auto_model.parameters():
                    param.requires_grad = False
        elif model_name == "t5-small":
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.transformer_encoder = T5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        if isinstance(self.transformer_encoder, SentenceTransformer):
            outputs = self.transformer_encoder.encode(
                x, show_progress_bar=False, device=device, convert_to_tensor=True
            )
        elif isinstance(self.transformer_encoder, T5ForConditionalGeneration):
            input_ids = self.tokenizer(
                x, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).input_ids.to(device)
            last_hidden_states = self.transformer_encoder.encoder(
                input_ids=input_ids
            ).last_hidden_state
            # Pooling strategy: here, we just take the representation of the first token
            outputs = last_hidden_states[:, 0]
        else:
            raise NotImplementedError

        return outputs


class ExpertEncoder(nn.Module):
    def __init__(
        self,
        expert_dim: int = 512,
        expert_num: int = 246,
        pretrained_embedding=None,
        pretrained_ids_to_tasks_names=None,
        tasks_names_to_ids=None,
    ):
        super().__init__()

        self.expert_embedding = nn.Embedding(expert_num, expert_dim)
        if pretrained_embedding is not None:
            for e, em in enumerate(pretrained_embedding):
                expert_name = pretrained_ids_to_tasks_names[e]
                index = tasks_names_to_ids[expert_name]
                self.expert_embedding.weight.data[index] = em

        # set the embedding trained
        self.expert_embedding.weight.requires_grad = True
        self.model = nn.Linear(expert_dim, expert_dim)

    def forward(self, expert_id):
        expert_embed = self.expert_embedding(expert_id)
        return self.model(expert_embed)


class ProjectionHead(nn.Module):
    def __init__(
        self, embedding_dim: int = 512, projection_dim: int = 512, dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        project = self.projection(x)
        x = self.layer_norm(project)
        return x


class CLIPRanker(AdapterRanker, EfficientCheckpointModule):
    def __init__(
        self,
        task_names,
        temperature: float = 0.07,
        text_embedding_dim: int = 384,
        expert_embedding_dim: int = 512,
        projection_dim: int = 512,
        encoder_model_name: str = "all-MiniLM-L6-v2",
        pretrained_embedding=None,
        pretrained_ids_to_tasks_names=None,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(task_names) > 0

        expert_names = task_names
        self.expert_num = len(expert_names)

        self.ids_to_tasks_names = {i: task for i, task in enumerate(expert_names)}
        self.tasks_names_to_ids = {task: i for i, task in enumerate(expert_names)}
        # initialize the§ text encoder and expert encoder
        self.text_encoder = TextEncoder(model_name=encoder_model_name)
        self.expert_encoder = ExpertEncoder(
            expert_dim=expert_embedding_dim,
            expert_num=self.expert_num,
            pretrained_embedding=pretrained_embedding,
            pretrained_ids_to_tasks_names=pretrained_ids_to_tasks_names,
            tasks_names_to_ids=self.tasks_names_to_ids,
        )
        self.projection_dim = projection_dim
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dim, projection_dim=projection_dim
        )
        self.expert_projection = ProjectionHead(
            embedding_dim=expert_embedding_dim, projection_dim=projection_dim
        )
        # mask for available tasks
        self.available_mask: torch.Tensor = torch.ones(self.expert_num)
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, batch):
        # gettng the expert and text features
        expert_ids = [
            self.tasks_names_to_ids[expert_name]
            for expert_name in batch["positive_expert_names"]
        ]
        expert_features = self.expert_encoder(torch.tensor(expert_ids).to(device))
        text_features = self.text_encoder(batch["sources_texts"])

        # Getting the expert and text embeddings with the same dimension
        expert_embeddings = self.expert_projection(expert_features)
        text_embeddings = self.text_projection(text_features)

        # l2 normalize the embeddings
        expert_embeddings = F.normalize(expert_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # calculate the loss
        logits = text_embeddings @ expert_embeddings.T / self.temperature

        # symmetric loss function
        labels = torch.arange(len(batch["positive_expert_names"])).to(self.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels) / 2

        return loss

    def set_available_tasks(self, available_tasks):
        """Set the available tasks for the classifier."""
        self.available_mask.fill_(0.0)

        for task in available_tasks:
            if "default" in task:
                continue
            if task in self.tasks_names_to_ids:
                self.available_mask[self.tasks_names_to_ids[task]] = 1.0

    def get_expert_embeddings(
        self,
    ):
        # we only need a num_experts x dimension matrix for the expert embeddings
        with torch.no_grad():
            expert_features = self.expert_encoder(
                torch.tensor(list(self.ids_to_tasks_names.keys())).to(device)
            )
            expert_embeddings = self.expert_projection(expert_features)
            return expert_embeddings

    def predict_task(self, query, n=1):
        # Getting the expert and text embeddings with the same dimension
        text_features = self.text_encoder(query)
        text_embeddings = self.text_projection(text_features)

        # get expert_embedding
        expert_embeddings = self.get_expert_embeddings()
        # l2 normalize the embeddings

        expert_embeddings = F.normalize(expert_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        # calculate the similarity
        logits = (
            (text_embeddings @ expert_embeddings.T / self.temperature).detach().cpu()
        )

        # masked the unavailable tasks
        if self.available_mask is not None:
            logits = logits + (1.0 - self.available_mask) * -100

        expert_indices = torch.topk(logits, k=n, dim=1)

        expert_prediction = [
            [self.ids_to_tasks_names[index.item()] for index in indices]
            for indices in expert_indices.indices
        ]
        expert_weights = [
            [weight.item() for weight in weights] for weights in expert_indices.values
        ]

        expert_weights = np.exp(np.array(expert_weights))
        expert_weights = expert_weights / expert_weights.sum(axis=1, keepdims=True)

        return expert_prediction, expert_weights.tolist()

    @torch.no_grad()
    def predict_batch(self, batch, n=1):
        text_features = self.text_encoder(batch["sources_texts"])
        # Getting the expert and text embeddings with the same dimension
        text_embeddings = self.text_projection(text_features)

        # get expert_embedding
        expert_embeddings = self.get_expert_embeddings()
        # l2 normalize the embeddings

        expert_embeddings = F.normalize(expert_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        # calculate the similarity
        logits = (
            (text_embeddings @ expert_embeddings.T / self.temperature).detach().cpu()
        )

        # masked the unavailable tasks
        if self.available_mask is not None:
            logits = logits + (1.0 - self.available_mask) * -100

        expert_indices = torch.topk(logits, k=n, dim=1)

        expert_prediction = [
            [self.ids_to_tasks_names[index.item()] for index in indices]
            for indices in expert_indices.indices
        ]
        expert_weights = [
            [weight.item() for weight in weights] for weights in expert_indices.values
        ]

        expert_weights = np.exp(np.array(expert_weights))
        expert_weights = expert_weights / expert_weights.sum(axis=1, keepdims=True)

        return expert_prediction, expert_weights.tolist()

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch["sources_texts"]),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(batch["sources_texts"]),
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CLIPTripletRanker(CLIPRanker):
    def get_ConsineSimilarity(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return torch.mean(x @ y.T)

    def forward(self, batch):
        # gettng the text features , positive, negatve expert
        text_features = self.text_encoder(batch["sources_texts"])
        positive_expert_ids = [
            self.tasks_names_to_ids[e] for e in batch["positive_expert_names"]
        ]
        negative_expert_ids = [
            self.tasks_names_to_ids[e] for e in batch["negative_expert_names"]
        ]
        positive_expert_features = self.expert_encoder(
            torch.tensor(positive_expert_ids).to(device)
        )
        negative_expert_features = self.expert_encoder(
            torch.tensor(negative_expert_ids).to(device)
        )

        # Getting the expert and text embeddings with the same dimension
        text_embeddings = self.text_projection(text_features)
        positive_expert_embeddings = self.expert_projection(positive_expert_features)
        negative_expert_embeddings = self.expert_projection(negative_expert_features)

        # l2 normalize the embeddings
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        positive_expert_embeddings = F.normalize(positive_expert_embeddings, dim=-1)
        negative_expert_embeddings = F.normalize(negative_expert_embeddings, dim=-1)

        # Compute distances (positive and negative scores)
        positive_score = torch.mean(text_embeddings @ positive_expert_embeddings.T)
        negative_score = torch.mean(text_embeddings @ negative_expert_embeddings.T)

        # calculate tripled loss
        loss = nn.TripletMarginLoss(margin=1.0)(
            text_embeddings, positive_expert_embeddings, negative_expert_embeddings
        )

        # log positive score and negative score
        self.log(
            "train/positive_score",
            positive_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )
        self.log(
            "train/negative_score",
            negative_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )

        return loss
