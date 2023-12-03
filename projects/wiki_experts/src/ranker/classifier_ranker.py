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
        self.text_encoder = self.text_encoder_init(
            requires_grad=kwargs["text_encoder_trained"]
        )
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

    def test_accuracy(self, dataset, model, fine_tune_task_name):
        if "flan" in dataset:
            classifier_config = ClassificationConfig(
                dataset=dataset,
                model=model,
                finetune_task_name=fine_tune_task_name,
            )
            datamodule = ClassificationDataModule(classifier_config)
        elif "adauni" in dataset:
            classifier_config = ClassificationConfig(
                dataset=dataset,
                model=model,
                finetune_task_name=fine_tune_task_name,
            )
            datamodule = ClassificationDataModuleFlatMultiTask(classifier_config)

        classifier = self.get_classifier()
        classifier.load_state_dict(torch.load(self.classifier_ckpt)["state_dict"])
        print("begin test")
        pbar = tqdm.tqdm(
            enumerate(datamodule.test_dataloader()),
            total=len(datamodule.test_dataloader()),
        )
        acc_all = []
        for _, batch in pbar:
            logits = classifier(batch["input"])
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(
                preds == batch["label"].clone().detach().to(device)
            ).item() / len(batch["label"])

            # predict expert names
            # print(
            #     "predict experts",
            #     [ids_to_tasks_names_ada[pred.item()] for pred in preds],
            # )
            acc_all.append(acc)
            pbar.set_description(f"Accuracy: {sum(acc_all)/len(acc_all)}")

        print(f"Accuracy: {sum(acc_all)/len(acc_all)}")

        # test mmlu accuracy
        from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig

        datamodule = MMLUDataModule(
            MMLUDataConfig(
                "mmlu",
                model="EleutherAI/gpt-neo-125m",
                model_family="gpt",
                train_batch_size=4,
                predict_batch_size=4,
                finetune_task_name=fine_tune_task_name,
            ),
            for_generation=True,
        )

        print("begin MMLU")
        pbar = tqdm.tqdm(
            enumerate(datamodule.test_dataloader()),
            total=len(datamodule.test_dataloader()),
        )

        for _, batch in pbar:
            logits = classifier(batch["sources_texts"])
            preds = torch.argmax(logits, dim=1)
            # acc = torch.sum(
            #     preds == batch["label"].clone().detach().to(device)
            # ).item() / len(batch["label"])

            # predict expert names
            print(
                "predict experts",
                [ids_to_tasks_names_ada[pred.item()] for pred in preds],
            )

    def predict_experts_using_classifier(self, input_texts):
        logits = self(input_texts)

        expert_indices = logits.argmax(dim=1).cpu()
        expert_prediction = [self.ids_to_tasks_names[i.item()] for i in expert_indices]
        return expert_prediction

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
    # from projects.wiki_experts.src.ranker.classification_module import (
    #     ClassificationDataModuleFlatMultiTask,
    #     ClassificationConfig,
    # )

    # from pytorch_lightning import Trainer

    # from projects.wiki_experts.src.config import ExpertConfig
    # from pytorch_lightning.callbacks import ModelCheckpoint
    # import os

    # config = ExpertConfig.parse()
    # dm = ClassificationDataModuleFlatMultiTask(
    #     ClassificationConfig(
    #         dataset=config.dataset,
    #         model=config.model,
    #         finetune_task_name=config.finetune_task_name,
    #         train_batch_size=config.train_batch_size,
    #         predict_batch_size=config.predict_batch_size,
    #     )
    # )

    # module = SentenceTransformerClassifier(num_labels=439)
    # module.to(device)
    # # for batch in dm.train_dataloader():
    # #     loss = model.training_step(batch, 0)
    # #     print(loss)
    # #     break
    # task_name = "adversarial_qa_dbert_answer_the_following_q"

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.getcwd() + f"/checkpoints/{task_name}",
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="val/loss",
    #     mode="min",
    #     filename=f"{task_name}" + "-{val/loss:.004f}",
    #     save_last=True,
    # )

    # trainer = Trainer(
    #     callbacks=[checkpoint_callback],
    #     max_epochs=3,
    #     max_steps=20,
    #     # val_check_interval=10,
    # )
    # trainer.fit(module, dm)
    model = SentenceTransformerClassifier(num_labels=439)
    model = model.from_pretrained(
        "zhan1993/adversarial_qa_dbert_answer_the_following_q_classifier"
    )
    model.to(device)

    predict_experts = model.predict_experts_using_classifier(
        [
            "if a horse at 2 years old has 3 legs, how many legs it has at 10 years old?",
            "if a horse at 2 years old has 3 legs, how many legs it has at 10 years old?",
        ]
    )
    print(predict_experts)
