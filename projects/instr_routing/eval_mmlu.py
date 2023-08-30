from transformers import AutoModelForCausalLM
import os
import torch


def eval_mmlu(
    config,
    model,
    data_dir=None,
    eval_batches=-1,
):
    from mttl.evaluators import MMLUEvaluator

    evaluator = MMLUEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
    )
    metrics = evaluator.evaluate(model, metric_per_task=True, eval_batches=eval_batches)
    torch.cuda.empty_cache()
    return metrics["exact_match"]["all"]


if __name__ == "__main__":
    from config import RoutingConfig
    from huggingface_hub import login
    
    login(token=os.environ["HF_TOKEN"])

    config = RoutingConfig.parse(extra_kwargs={"eval_superni": False})

    config.model = "meta-llama/Llama-2-7b-hf"
    config.load_in_8bit = True
    config.model_family = "gpt"
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 4096
    config.max_output_length = 5

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        load_in_8bit=config.load_in_8bit,
        device_map="auto"
    )
    print(eval_mmlu(config, model, eval_batches=200))
