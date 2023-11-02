from transformers import AutoModelForCausalLM
import os
import torch


def eval_mmlu(
    config,
    model,
    subsample=-1,
):
    from mttl.evaluators import MMLUEvaluator

    evaluator = MMLUEvaluator(
        config,
    )
    metrics = evaluator.evaluate(model, subsample=subsample)
    torch.cuda.empty_cache()
    return metrics


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from config import RoutingConfig
    from huggingface_hub import login

    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])

    config = RoutingConfig.parse(extra_kwargs={"eval_superni": False})

    config.model = "EleutherAI/gpt-neo-125m"
    config.load_in_8bit = True
    config.model_family = "gpt"
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 4096
    config.max_output_length = 5

    model = AutoModelForCausalLM.from_pretrained(
        config.model, load_in_8bit=config.load_in_8bit, device_map="auto"
    )
    print(eval_mmlu(config, model, subsample=10))
