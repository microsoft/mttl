from projects.wiki_experts.src.ranker.baseline_rankers import KATERouter
from projects.wiki_experts.src.ranker.classifier_ranker import (
    SentenceTransformerClassifier,
)
from projects.wiki_experts.src.ranker.clip_ranker import CLIPRanker

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdapterRankerHelper:
    @staticmethod
    def get_ranker_instance(
        ranker_model,
        ranker_path,
    ):
        if ranker_model == "clip":
            model = CLIPRanker().from_pretrained(ranker_path).to(device)
            return model
        elif ranker_model == "classifier":
            model = SentenceTransformerClassifier.from_pretrained(ranker_path).to(
                device
            )
            return model
        elif ranker_model == "kate":
            model = KATERouter.from_pretrained(ranker_path)
            return model
        else:
            raise ValueError(f"Unknown retrieval model: {ranker_model}")


if __name__ == "__main__":
    # config = ExpertConfig.parse()
    expert_ranker = AdapterRankerHelper(
        retrieval_model="classifier",
        model_path="zhan1993/adversarial_qa_dbert_answer_the_following_q_classifier",
    )
    print(
        expert_ranker.get_predict_experts(
            ["what is the capital of france?", "what is the capital of france?"]
        )
    )
    expert_ranker = AdapterRankerHelper(
        retrieval_model="clip",
        model_path="zhan1993/clip_ranker_debug",
    )
    print(
        expert_ranker.get_predict_experts(
            ["what is the capital of france?", "what is the capital of france?"]
        )
    )
    # input_text = (
    #     "if a horse at 2 years old has 3 legs, how many legs it has at 10 years old?"
    # )
    # logits = classifier([input_text])

    # expert_indices = logits.argmax(dim=1).cpu()
    # expert_prediction = [
    #     expert_ranker.ids_to_tasks_names[i.item()] for i in expert_indices
    # ]

    # print(expert_prediction)
    # expert_ranker.compute_expert_similarity()
    # expert_ranker.get_predict_retrieval()
    # expert_ranker.test_accuracy(config.dataset, config.model, config.finetune_task_name)
