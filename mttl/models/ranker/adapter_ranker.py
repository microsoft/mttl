from abc import ABC, abstractmethod

import torch


class AdapterRanker(ABC):
    @abstractmethod
    def predict_batch(self, batch, n=1):
        """Predicts the top n tasks for each input in the batch."""
        pass

    @abstractmethod
    def predict_task(self, query, n=1):
        """Predicts the top n tasks for the input query."""
        pass


class AdapterRankerHelper:
    @staticmethod
    def get_ranker_instance(ranker_model, ranker_path, device="cuda"):
        from mttl.models.ranker.baseline_rankers import KATERanker
        from mttl.models.ranker.classifier_ranker import (
            ClusterPredictor,
            SentenceTransformerClassifier,
        )
        from mttl.models.ranker.clip_ranker import CLIPRanker, CLIPTripletRanker

        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"

        if ranker_model == "clip":
            model = CLIPRanker.from_pretrained(ranker_path).to(device)
            return model
        elif ranker_model == "clip_triplet":
            model = CLIPTripletRanker.from_pretrained(ranker_path).to(device)
            return model
        elif ranker_model == "classifier":
            model = SentenceTransformerClassifier.from_pretrained(ranker_path).to(
                device
            )
            return model
        elif ranker_model == "kate":
            model = KATERanker.from_pretrained(ranker_path)
            return model
        elif ranker_model == "cluster_predictor":
            model = ClusterPredictor.from_pretrained(ranker_path)
            return model
        else:
            raise ValueError(f"Unknown retrieval model: {ranker_model}")
