from projects.wiki_experts.src.ranker.baseline_rankers import KATERouter
from projects.wiki_experts.src.ranker.classifier_ranker import (
    SentenceTransformerClassifier,
)
from projects.wiki_experts.src.ranker.clip_ranker import CLIPRanker

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdapterRankerHelper:
    def __init__(
        self,
        ranker_model,
        ranker_path,
    ):
        self.ranker_model = ranker_model
        self.ranker_path = ranker_path
        self.ranker = self.get_ranker_instance()

    def get_ranker_instance(
        self,
    ):
        if self.ranker_model == "clip":
            model = CLIPRanker().from_pretrained(self.ranker_path).to(device)
            return model
        elif self.ranker_model == "classifier":
            model = (
                SentenceTransformerClassifier()
                .from_pretrained(self.ranker_path)
                .to(device)
            )
            return model
        elif self.ranker_model == "kate":
            model = KATERouter.from_pretrained(self.ranker_path)
            return model
        else:
            raise ValueError(f"Unknown retrieval model: {self.ranker_model}")

    def predict_task(self, query):
        return self.ranker.predict_task(query)

    def predict_batch(self, batch):
        return self.ranker.predict_batch(batch)


if __name__ == "__main__":
    # config = ExpertConfig.parse()

    expert_ranker = AdapterRankerHelper(
        ranker_model="clip",
        ranker_path="zhan1993/clip_ranker_debug",
    )

    batch = [
        "what is the capital of france?",
        "what is the capital of france?",
    ]
    print(
        expert_ranker.predict_task(
            ["what is the capital of france?", "what is the capital of france?"]
        )
    )
