import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from projects.wiki_experts.src.ranker.experts_ranker import ExpertsRanker
from projects.wiki_experts.src.config import (
    tasks_names_to_ids_ada,
    tasks_names_to_ids,
    ids_to_tasks_names,
    ids_to_tasks_names_ada,
)


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
        num_labels=439,
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
            self.ids_to_tasks_names = ids_to_tasks_names_ada
        else:
            self.ids_to_tasks_names = ids_to_tasks_names
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
        # change the "niv2_misc." to "niv2_misc"
        for i in range(len(task_name)):
            if task_name[i] == "niv2_misc.":
                task_name[i] = "niv2_misc"
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

    def test_accuracy(self, dataset, data_model, fine_tune_task_name):
        from projects.wiki_experts.src.ranker.classification_module import (
            ClassificationConfig,
            ClassificationDataModuleFlatMultiTask,
        )
        from tqdm import tqdm
        import numpy as np

        classifier_config = ClassificationConfig(
            dataset=dataset,
            model=data_model,
            finetune_task_name=fine_tune_task_name,
        )
        datamodule = ClassificationDataModuleFlatMultiTask(classifier_config)

        pbar = tqdm(
            enumerate(datamodule.test_dataloader()),
            total=len(datamodule.test_dataloader()),
        )
        acc_all = []
        for _, batch in pbar:
            preds = self.predict_experts_using_classifier(batch["input"])

            acc = np.sum(np.array(preds) == np.array(batch["task_name"])) / len(preds)
            acc_all.append(acc)
            pbar.set_description(f"Accuracy: {sum(acc_all)/len(acc_all)}")

        print(f"Accuracy: {sum(acc_all)/len(acc_all)}")

    def predict_experts_using_classifier(self, input_texts):
        logits = self(input_texts)

        expert_indices = logits.argmax(dim=1).cpu()
        expert_prediction = [self.ids_to_tasks_names[i.item()] for i in expert_indices]
        return expert_prediction

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


if __name__ == "__main__":
    model = SentenceTransformerClassifier()
    model = model.from_pretrained("zhan1993/classifier_ranker")
    model.to(device)
    model.test_accuracy(
        "sordonia/adauni-v1-flat",
        "EleutherAI/gpt-neo-125m",
        fine_tune_task_name="astronomy",
    )

    # from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
    from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig
    from projects.wiki_experts.src.expert_model import MultiExpertModelRanker
    from projects.wiki_experts.src.ranker.adapter_ranker import AdapterRankerHelper

    finetune_task_name = "astronomy"
    data_module = MMLUDataModule(
        MMLUDataConfig(
            "mmlu",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            train_batch_size=4,
            predict_batch_size=4,
            finetune_task_name=finetune_task_name,
        ),
        for_generation=True,
    )

    dm = data_module.val_dataloader()
    batch = next(iter(dm))

    predict_experts = model.predict_experts_using_classifier(batch["sources_texts"])
    print(predict_experts)

    predict_scores = model.predict_scores_using_classifier(batch["sources_texts"])
    print(predict_scores)

    # from projects.wiki_experts.src.config import ExpertConfig

    # config = ExpertConfig()
    # config.routing = "retrieval"
    # config.model = "EleutherAI/gpt-neo-125m"
    # config.retrieval_model = "classifier"
    # config.expert_model_path = "zhan1993/classifier_ranker"

    # model = AdapterRankerHelper(config.retrieval_model, config.expert_model_path)
    # prediction_experts = model.ranker.predict_experts_using_classifier(sources_texts)
    # print(prediction_experts)
