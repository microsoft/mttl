# implements the CLIPRanker class
from sentence_transformers import SentenceTransformer
from projects.wiki_experts.src.ranker.clip_data_module import (
    CLIPExpertsDatamodule,
    CLIPExpertsConfig,
)
from projects.wiki_experts.src.config import ids_to_tasks_names
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os

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
    def __init__(self, expert_dim: int = 512):
        super().__init__()

        self.expert_embedding = nn.Embedding(246, expert_dim)
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


class CLIPRanker(pl.LightningModule):
    def __init__(
        self,
        temperature: float = 0.07,
        text_embedding_dim: int = 384,
        expert_embedding_dim: int = 512,
    ):
        super().__init__()

        self.text_encoder = TextEncoder()
        self.expert_encoder = ExpertEncoder()

        self.text_projection = ProjectionHead(embedding_dim=384)
        self.expert_projection = ProjectionHead(embedding_dim=expert_embedding_dim)
        self.temperature = temperature

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
        # we only need a 246 x 512 matrix for the expert embeddings
        with torch.no_grad():
            expert_features = self.expert_encoder(
                torch.tensor(list(ids_to_tasks_names.keys())).to(device)
            )
            expert_embeddings.append(self.expert_projection(expert_features))
            return torch.cat(expert_embeddings)

    def predict_experts_using_clip(self, batch, expert_embeddings, n=1):
        if "input_texts" in batch:
            text_features = self.text_encoder(batch["input_texts"])
        else:
            text_features = self.text_encoder(batch["sources_texts"])

        # Getting the expert and text embeddings with the same dimension
        text_embeddings = self.text_projection(text_features)

        # l2 normalize the embeddings
        expert_embeddings = F.normalize(expert_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        # calculate the similarity
        dot_similarity = text_embeddings @ expert_embeddings.T / self.temperature

        values, indices = torch.topk(dot_similarity.squeeze(0), n)
        # now we only selet the top n experts
        matches = [ids_to_tasks_names[idx[0].item()] for idx in indices]

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


if __name__ == "__main__":
    model = CLIPRanker()
    model.to(device)

    model.load_state_dict(torch.load(os.environ["CLIP_CKPT"])["state_dict"])

    # test the model
    dataconfig = CLIPExpertsConfig(model="EleutherAI/gpt-neo-125m")
    dm = CLIPExpertsDatamodule(dataconfig)

    # get the expert embeddings
    expert_embeddings = model.get_expert_embeddings()
    print(expert_embeddings.shape)
    # get the top 5 experts for each example in the test set
    for batch in dm.test_dataloader(subsample=10):
        print(model.predict_experts_using_clip(batch, expert_embeddings))
        break
