# load the ranker and predict the experts we use
from projects.wiki_experts.classification_module import ClassificationDataModule
from projects.wiki_experts.models.classifer_ranker import Classifier
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from projects.wiki_experts.src.config import ExpertConfig
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExpertRanker:
    def __init__(self, num_labels, classifer_repo_id, predict_batch_size=32):
        self.num_labels = num_labels
        self.classifer_ckpt = os.environ.get("CLASSIFIER_CKPT")
        self.classifer_repo_id = classifer_repo_id
        self.predict_batch_size = predict_batch_size

    # get a single instance of the classifier
    def get_classifier(
        self,
    ):
        if self.classifer_ckpt is None:
            self.classifer_ckpt = hf_hub_download(
                repo_id=self.classifer_repo_id,
                filename="checkpoint.ckpt",
                repo_type="model",
            )
        print(f"Downloaded the classifier from {self.classifer_ckpt}")
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        classifier = Classifier(text_encoder, self.num_labels).to(device)
        classifier.load_state_dict(torch.load(self.classifer_ckpt)["state_dict"])
        return classifier

    def test_accuracy(self):
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        datamodule = ClassificationDataModule(batch_size=self.predict_batch_size)
        datamodule.setup("test")
        classifier = Classifier(text_encoder, self.num_labels).to(device)
        classifier.load_state_dict(torch.load(self.classifer_ckpt)["state_dict"])
        print("begin test")
        acc_all = []
        for batch in datamodule.test_dataloader():
            logits = classifier(batch["input"])
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == batch["label"]).item() / len(batch["label"])
            acc_all.append(acc)
        print(f"Accuracy: {sum(acc_all)/len(acc_all)}")


if __name__ == "__main__":
    config = ExpertConfig.parse()
    expert_ranker = ExpertRanker(config.num_labels, config.classifer_repo_id)
    expert_ranker.test_accuracy()
