# implements the CLIPRanker class
from sentence_transformers import SentenceTransformer
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os

from projects.wiki_experts.src.ranker.adapter_ranker import AdapterRanker
from mttl.models.utils import EfficientCheckpointModule


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    def __init__(
        self,
        trainable: bool = False,
    ):
        super().__init__()
        self.transformer_encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )  # You need to define your text encoder

        # frozen the transformer parameters
        auto_model = self.transformer_encoder._first_module().auto_model
        if not trainable:
            for param in auto_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        outputs = torch.tensor(self.transformer_encoder.encode(x)).to(device)
        return outputs


class ExpertEncoder(nn.Module):
    def __init__(self, expert_dim: int = 512, expert_num: int = 246):
        super().__init__()

        self.expert_embedding = nn.Embedding(expert_num, expert_dim)
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
            nn.Dropout(dropout),
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
        expert_num: int = 246,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_encoder = TextEncoder()
        self.expert_encoder = ExpertEncoder(
            expert_dim=expert_embedding_dim,
            expert_num=expert_num,
        )
        self.ids_to_tasks_names = task_names
        self.expert_num = expert_num
        self.text_projection = ProjectionHead(embedding_dim=384)
        self.expert_projection = ProjectionHead(embedding_dim=expert_embedding_dim)
        self.temperature = temperature
        self.save_hyperparameters()

    def forward(self, batch):
        # gettng the expert and text features
        expert_features = self.expert_encoder(batch["expert_id"].to(device))
        text_features = self.text_encoder(batch["input_texts"])

        # Getting the expert and text embeddings with the same dimension
        expert_embeddings = self.expert_projection(expert_features)
        text_embeddings = self.text_projection(text_features)

        # l2 normalize the embeddings
        expert_embeddings = F.normalize(expert_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        # calculate the loss
        logits = text_embeddings @ expert_embeddings.T / self.temperature

        # symmetric loss function
        labels = torch.arange(len(batch["expert_id"])).to(self.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels) / 2

        return loss

    def get_expert_embeddings(
        self,
    ):
        expert_embeddings = []
        # we only need a num_experts x dimension matrix for the expert embeddings
        with torch.no_grad():
            expert_features = self.expert_encoder(
                torch.tensor(list(self.ids_to_tasks_names.keys())).to(device)
            )
            expert_embeddings.append(self.expert_projection(expert_features))
            return torch.cat(expert_embeddings)

    def predict_experts_using_clip(self, input_texts, top_n=1):
        text_features = self.text_encoder(input_texts)
        # Getting the expert and text embeddings with the same dimension
        text_embeddings = self.text_projection(text_features)
        # get expert_embedding
        expert_embeddings = self.get_expert_embeddings()

        # l2 normalize the embeddings
        expert_embeddings = F.normalize(expert_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        # calculate the similarity
        dot_similarity = text_embeddings @ expert_embeddings.T / self.temperature

        values, indices = torch.topk(dot_similarity.squeeze(0), top_n)
        # now we only selet the top n experts

        matches = []
        if top_n == 1:
            matches = [self.ids_to_tasks_names[idx.item()] for idx in indices]
            return matches
        for item in indices:
            m = []
            for idx in item:
                m.append(self.ids_to_tasks_names[idx.item()])
            m.append(self.ids_to_tasks_names[item.item()])
            matches.append(m)

        return matches

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log(
            "val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CLIPTripletRanker(CLIPRanker):
    def __init__(
        self,
        temperature: float = 0.07,
        text_embedding_dim: int = 384,
        expert_embedding_dim: int = 512,
        expert_num: int = 246,
    ):
        super().__init__(
            temperature=temperature,
            text_embedding_dim=text_embedding_dim,
            expert_embedding_dim=expert_embedding_dim,
            expert_num=expert_num,
        )

    def forward(self, batch):
        # gettng the text features , positive, negatve expert
        text_features = self.text_encoder(batch["input_texts"])
        positive_expert_features = self.expert_encoder(
            batch["positive_expert_id"].to(device)
        )
        negative_expert_features = self.expert_encoder(
            batch["negative_expert_id"].to(device)
        )

        # Getting the expert and text embeddings with the same dimension
        text_embeddings = self.text_projection(text_features)
        positive_expert_embeddings = self.expert_projection(positive_expert_features)
        negative_expert_embeddings = self.expert_projection(negative_expert_features)

        # # l2 normalize the embeddings
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        positive_expert_embeddings = F.normalize(positive_expert_embeddings, dim=-1)
        negative_expert_embeddings = F.normalize(negative_expert_embeddings, dim=-1)

        # Compute distances (positive and negative scores)
        positive_score = torch.mean(
            torch.nn.functional.pairwise_distance(
                text_embeddings, positive_expert_embeddings, p=2
            )
        )
        negative_score = torch.mean(
            torch.nn.functional.pairwise_distance(
                text_embeddings, negative_expert_embeddings, p=2
            )
        )

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
        )
        self.log(
            "train/negative_score",
            negative_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss
