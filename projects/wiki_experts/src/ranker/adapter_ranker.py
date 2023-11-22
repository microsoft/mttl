# load the ranker and predict the experts we use
from projects.wiki_experts.src.ranker.classification_module import (
    ClassificationDataModule,
    ClassificationConfig,
)
from projects.wiki_experts.src.ranker.classifier_ranker import Classifier
from projects.wiki_experts.src.ranker.clip_ranker import CLIPRanker
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from projects.wiki_experts.src.config import ExpertConfig
import torch
import os
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExpertRanker:
    def __init__(self, num_labels, classifier_repo_id, predict_batch_size=32):
        self.num_labels = num_labels
        self.classifier_ckpt = os.environ.get("CLASSIFIER_CKPT")
        self.classifier_repo_id = classifier_repo_id
        self.predict_batch_size = predict_batch_size

    def get_clip_ranker(self):
        self.clip_ckpt = os.environ.get("CLIP_CKPT", None)
        if self.clip_ckpt is None:
            raise ValueError(
                "Please provide a clip_ckpt or set the CLIP_CKPT environment variable"
            )
        clip_ranker = CLIPRanker().to(device)
        clip_ranker.load_state_dict(torch.load(self.clip_ckpt)["state_dict"])
        return clip_ranker

    # get a single instance of the classifier
    def get_classifier(
        self,
    ):
        if self.classifier_ckpt is None:
            if self.classifier_repo_id is None:
                raise ValueError(
                    "Please provide a classifier_repo_id or set the classifier_CKPT environment variable"
                )
            self.classifier_ckpt = hf_hub_download(
                repo_id=self.classifier_repo_id,
                filename="checkpoint.ckpt",
                repo_type="model",
            )
        print(f"Downloaded the classifier from {self.classifier_ckpt}")
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        classifier = classifier(text_encoder, self.num_labels).to(device)
        classifier.load_state_dict(
            torch.load(self.classifier_ckpt, map_location=device)["state_dict"]
        )
        return classifier

    def test_accuracy(self, dataset, model, fine_tune_task_name):
        classifier_config = ClassificationConfig(
            dataset=dataset,
            model=model,
            finetune_task_name=fine_tune_task_name,
        )
        datamodule = ClassificationDataModule(classifier_config)
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
            acc_all.append(acc)
            pbar.set_description(f"Accuracy: {sum(acc_all)/len(acc_all)}")

        print(f"Accuracy: {sum(acc_all)/len(acc_all)}")


if __name__ == "__main__":
    config = ExpertConfig.parse()
    expert_ranker = ExpertRanker(config.num_labels, config.classifier_repo_id)
    expert_ranker.test_accuracy(config.dataset, config.model, config.finetune_task_name)
