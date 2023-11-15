# load the ranker and predict the experts we use
from classification_module import ClassificationDataModule
from models.classifer_ranker import Classifier
from sentence_transformers import SentenceTransformer
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classfier_ckpt = os.environ.get("CLASSIFIER_CKPT")


# get a single instance of the classifier
def get_classifier():
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    classifier = Classifier(text_encoder, 246).to(device)
    classifier.load_state_dict(torch.load(classfier_ckpt)["state_dict"])
    return classifier


def test_accuracy():
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    datamodule = ClassificationDataModule(batch_size=100)
    datamodule.setup("test")

    classifier = Classifier(text_encoder, 246).to(device)
    classifier.load_state_dict(torch.load(classfier_ckpt)["state_dict"])
    print("begin test")
    acc_all = []
    for batch in datamodule.test_dataloader():
        logits = classifier(batch["input"])
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == batch["label"]).item() / len(batch["label"])
        acc_all.append(acc)
    print(f"Accuracy: {sum(acc_all)/len(acc_all)}")


def test_dataset():
    datamodule = ClassificationDataModule(batch_size=1024)
    datamodule.setup("test")

    for batch in datamodule.test_dataloader():
        print(batch["label"])


if __name__ == "__main__":
    test_accuracy()
