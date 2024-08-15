import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from mttl.models.library.expert_library import DatasetLibrary
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
        if model_name == "t5-small":
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.transformer_encoder = T5ForConditionalGeneration.from_pretrained(
                model_name, return_dict=True
            )
        else:
            # consider it as a sentence transformer model
            if "/" in model_name:
                model_name = model_name.split("/")[1]
            self.transformer_encoder = SentenceTransformer(model_name)
            # freeze the transformer parameters
            auto_model = self.transformer_encoder._first_module().auto_model
            if not trainable:
                for param in auto_model.parameters():
                    param.requires_grad = False

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


class SentenceTransformerClassifier(AdapterRanker, EfficientCheckpointModule):
    # define the classifier, the x is the input, the task_id or expert_id is the label
    def __init__(
        self,
        task_names,
        hidden_size=768,
        transformer_embed_dim=384,
        temperature=1,
        encoder_model_name="all-MiniLM-L6-v2",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_encoder = TextEncoder(model_name=encoder_model_name)
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
        text_output = self.text_encoder(x)
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
        logits = self.forward(query).detach().cpu()

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


class ClassifierSmooth(SentenceTransformerClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dataset = DatasetLibrary.pull_dataset("zhan1993/transfer_matrix_v3")
        self.transfer_matrix_df = dataset["train"].to_pandas()
        self.transfer_matrix_df.set_index(["expert_name", "task_eval_on"], inplace=True)
        self.task_names_to_distribution = {}
        for task_name in self.task_names_to_ids:
            # initialize the distribution
            self.task_names_to_distribution[task_name] = np.ones(
                len(self.task_names_to_ids)
            )
            for expert_name in self.task_names_to_ids:
                # we get each expert score for each task
                if (expert_name, task_name) in self.transfer_matrix_df.index:
                    loss_score = self.transfer_matrix_df.loc[
                        (expert_name, task_name), "score"
                    ]
                else:
                    loss_score = 100
                self.task_names_to_distribution[task_name][
                    self.task_names_to_ids[expert_name]
                ] = loss_score

    def get_task_names_distribution(self, task_names):
        """
        Converts a list of task names to their corresponding scores. Here
        the scores are from the transfer matrix distribution.

        Args:
            task_names (list): A list of task names.
            [batch(task_names)]
        Returns:
            batch [batch,N] N is the number of available experts.
        """
        loss_scores = []
        for task_name in task_names:
            assert task_name in self.task_names_to_ids
            loss_scores.append(self.task_names_to_distribution[task_name])
        batch_score = torch.tensor(loss_scores).to(device)
        return batch_score

    def get_expert_distribution(self, batch):
        # get the expert distribution for the batch. [batch, N]
        expert_distribution = self.get_task_names_distribution(batch["task_names"])
        return expert_distribution

    def training_step(self, batch, batch_idx):
        logits = self(batch["sources_texts"])
        scores = self.get_expert_distribution(batch)
        scores = -scores / 0.1
        log_scores = torch.log_softmax(scores, -1)
        scores = torch.softmax(scores, -1)  # note that scores are loss scores
        # logits = logits / 0.1
        probs = torch.log_softmax(logits, -1)
        loss = torch.mean(-(scores * probs - scores * log_scores).sum(1))
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
        logits = self(batch["sources_texts"])
        scores = self.get_expert_distribution(batch)
        scores = -scores / 0.1
        log_scores = torch.log_softmax(scores, -1)
        scores = torch.softmax(scores, -1)  # note that scores are loss scores
        # logits = logits / 0.1
        probs = torch.log_softmax(logits, -1)
        loss = torch.mean(-(scores * probs - scores * log_scores).sum(1))
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch["sources_texts"]),
        )


class ClusterPredictor(SentenceTransformerClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.no_grad()
    def init_clusters(self, hf_lib_id):
        from mttl.models.library.expert_library import HFExpertLibrary

        library = HFExpertLibrary(hf_lib_id)
        self.cluster_names = {}
        self.cluster_names_to_expert_ids = {}
        assert library is not None
        # load the cluster names and experts_names from the library

        for expert in library.keys():
            expert_dump = library.get_expert(expert)
            cluster_name = expert_dump.expert_info.expert_name
            expert_task_name = expert_dump.expert_info.expert_task_name
            self.cluster_names[cluster_name] = expert_task_name
        for cluster_name in self.cluster_names:
            self.cluster_names_to_expert_ids[cluster_name] = [
                self.task_names_to_ids[task_name]
                for task_name in self.cluster_names[cluster_name].split(",")
            ]

        self.cluster_names_to_ids = {
            cluster_name: i for i, cluster_name in enumerate(self.cluster_names)
        }

        self.ids_to_cluster_names = {
            i: cluster_name for i, cluster_name in enumerate(self.cluster_names)
        }

    @torch.no_grad()
    def predict_task(self, query, n=1):
        logits = self(query).detach().cpu()

        # softmax
        logits = torch.softmax(logits, dim=-1)

        # get the cluster distribution
        cluster_distribution = torch.zeros(logits.shape[0], len(self.cluster_names))
        for cluster_name in self.cluster_names_to_ids:
            cluster_distribution[:, self.cluster_names_to_ids[cluster_name]] = (
                torch.sum(
                    logits[:, self.cluster_names_to_expert_ids[cluster_name]], dim=-1
                )
            )
        # get the topk clusters
        cluster_indices = torch.topk(cluster_distribution, k=n, dim=1)

        cluster_prediction = [
            [self.ids_to_cluster_names[index.item()] for index in indices]
            for indices in cluster_indices.indices
        ]

        cluster_weights = [
            [weight.item() for weight in weights] for weights in cluster_indices.values
        ]
        # increate the entropy of the weights
        cluster_weights = np.array(cluster_weights) / self.temperature

        cluster_weights = np.exp(np.array(cluster_weights))
        cluster_weights = cluster_weights / cluster_weights.sum(axis=1, keepdims=True)

        return cluster_prediction, cluster_weights.tolist()

    @torch.no_grad()
    def predict_batch(self, batch, n=1):
        logits = self(batch["sources_texts"]).detach().cpu()

        # softmax
        logits = torch.softmax(logits, dim=-1)

        # get the cluster distribution
        cluster_distribution = torch.zeros(logits.shape[0], len(self.cluster_names))
        for cluster_name in self.cluster_names_to_ids:
            cluster_distribution[:, self.cluster_names_to_ids[cluster_name]] = (
                torch.sum(
                    logits[:, self.cluster_names_to_expert_ids[cluster_name]], dim=-1
                )
            )

        # get the topk clusters
        cluster_indices = torch.topk(cluster_distribution, k=n, dim=1)

        cluster_prediction = [
            [self.ids_to_cluster_names[index.item()] for index in indices]
            for indices in cluster_indices.indices
        ]

        cluster_weights = [
            [weight.item() for weight in weights] for weights in cluster_indices.values
        ]
        # increate the entropy of the weights
        cluster_weights = np.array(cluster_weights) / self.temperature

        cluster_weights = np.exp(np.array(cluster_weights))
        cluster_weights = cluster_weights / cluster_weights.sum(axis=1, keepdims=True)

        return cluster_prediction, cluster_weights.tolist()
