import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from projects.wiki_experts.src.ranker.experts_ranker import ExpertsRanker


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
        task_names=[],
        hidden_size=768,
        transformer_embed_dim=384,
        freeze_text_encoder=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_encoder = self.text_encoder_init(False)
        self.labels_texts = task_names
        self.task_names_to_ids = {task: i for i, task in enumerate(task_names)}
        self.num_labels = len(task_names)

        # linear text encoder
        self.text_projecter = nn.Linear(transformer_embed_dim, hidden_size)
        self.out_projecter = nn.Linear(hidden_size, self.num_labels)
        self.save_hyperparameters(ignore=["text_encoder"])

    def forward(self, x):
        # Encode the text input
        text_output = torch.tensor(self.text_encoder.encode(x)).to(device)
        # conver the text output to hidden vector
        text_output_projecter = self.text_projecter(text_output)
        # Calculate the logits
        logits = self.out_projecter(text_output_projecter)
        return logits

    def predict_task(self, query):
        assert type(query) == str
        probs = torch.softmax(self.forward(query), -1)
        results = torch.argsort(probs, dim=1, descending=True)
        return [self.labels_texts[int(i)] for i in results[0][:3]], [
            probs[0][i] for i in results[0][:3]
        ]

    def predict_batch(self, batch):
        query = batch["sources_texts"]
        probs = torch.softmax(self.forward(query), -1)
        top_tasks, top_weights = [], []
        for _, probs_ in enumerate(probs):
            best_index = probs_.argsort(descending=True)
            top_tasks.append([self.labels_texts[i] for i in best_index[:3]])
            top_weights.append([probs_[i] for i in best_index[:3]])
        return top_tasks, top_weights

    def text_encoder_init(self, requires_grad=False):
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

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

    def predict_scores_using_classifier(self, input_texts):
        logits = self(input_texts)

        softmax = nn.Softmax(dim=1)
        logits = softmax(logits)

        max_scores = logits.max(dim=1).values.cpu().detach().numpy()
        return max_scores

    def compute_expert_similarity(self):
        import numpy as np
        import json

        classifier = self.get_classifier()
        classifier.load_state_dict(torch.load(self.classifier_ckpt)["state_dict"])
        sim = classifier.classifier.weight @ classifier.classifier.weight.T

        # get the top 5 experts for each expert
        top_k = 5
        top_k_sim, top_k_sim_indices = torch.topk(sim, top_k, dim=1)

        # convert the tensor to cpu
        top_k_sim_indices = top_k_sim_indices.cpu()

        fout = open("top_5_random_5.jsonl", "w")
        # print the top k experts for each expert
        for i in range(top_k_sim_indices.shape[0]):
            candidate_experts = []
            for j in range(top_k_sim_indices.shape[1]):
                # add a most similar one
                candidate_experts.append(
                    self.ids_to_tasks_names[top_k_sim_indices[i][j].item()]
                )
                # add a random one
                candidate_experts.append(
                    self.ids_to_tasks_names[np.random.randint(0, self.num_labels)]
                )
            fout.write(
                json.dumps(
                    {
                        "task": self.ids_to_tasks_names[i],
                        "candidate_experts": candidate_experts,
                    }
                )
                + "\n"
            )

        # draw the similarity matrix using headmap

        import matplotlib.pyplot as plt

        plt.matshow(sim.detach().cpu().numpy())
        plt.colorbar()
        plt.show()
        plt.savefig("similarity_matrix.png")
