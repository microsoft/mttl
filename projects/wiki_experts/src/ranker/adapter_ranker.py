# load the ranker and predict the experts we use
from projects.wiki_experts.src.ranker.classification_module import (
    ClassificationDataModule,
    ClassificationConfig,
)
from projects.wiki_experts.src.ranker.classifer_ranker import Classifer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from projects.wiki_experts.src.config import ExpertConfig
import torch
import os
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExpertRanker:
    def __init__(self, num_labels, classifer_repo_id, predict_batch_size=32):
        self.num_labels = num_labels
        self.classifer_ckpt = os.environ.get("CLASSIFER_CKPT")
        self.classifer_repo_id = classifer_repo_id
        self.predict_batch_size = predict_batch_size

    # get a single instance of the classifer
    def get_classifer(
        self,
    ):
        if self.classifer_ckpt is None:
            if self.classifer_repo_id is None:
                raise ValueError(
                    "Please provide a classifer_repo_id or set the CLASSIFER_CKPT environment variable"
                )
            self.classifer_ckpt = hf_hub_download(
                repo_id=self.classifer_repo_id,
                filename="checkpoint.ckpt",
                repo_type="model",
            )
        print(f"Downloaded the classifer from {self.classifer_ckpt}")
        text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        classifer = Classifer(text_encoder, self.num_labels).to(device)
        classifer.load_state_dict(torch.load(self.classifer_ckpt)["state_dict"])
        return classifer

    def test_accuracy(self, dataset, model, fine_tune_task_name):
        classifer_config = ClassificationConfig(
            dataset=dataset,
            model=model,
            finetune_task_name=fine_tune_task_name,
        )
        datamodule = ClassificationDataModule(classifer_config)
        classifer = self.get_classifer()
        classifer.load_state_dict(torch.load(self.classifer_ckpt)["state_dict"])
        print("begin test")
        pbar = tqdm.tqdm(
            enumerate(datamodule.test_dataloader()),
            total=len(datamodule.test_dataloader()),
        )
        acc_all = []
        for _, batch in pbar:
            logits = classifer(batch["input"])
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(
                preds == batch["label"].clone().detach().to(device)
            ).item() / len(batch["label"])
            acc_all.append(acc)
            pbar.set_description(f"Accuracy: {sum(acc_all)/len(acc_all)}")

        print(f"Accuracy: {sum(acc_all)/len(acc_all)}")


if __name__ == "__main__":
    config = ExpertConfig.parse()
    expert_ranker = ExpertRanker(config.num_labels, config.classifer_repo_id)
    expert_ranker.test_accuracy(config.dataset, config.model, config.finetune_task_name)
