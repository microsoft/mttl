from transformers import AutoModelForCausalLM
import os
import torch


def eval_mmlu(
    config,
    model,
    data_dir=None,
    eval_batches=-1,
):
    from mttl.models.mmlu_evaluator import MMLUEvaluator

    evaluator = MMLUEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
    )
    metrics = evaluator.evaluate(model, metric_per_task=True, eval_batches=eval_batches)
    torch.cuda.empty_cache()
    return metrics["exact_match"]["all"]


if __name__ == "__main__":
    from projects.instr_routing.finetune_llama import RoutingConfig

    config = RoutingConfig.parse()

    config.model = "uoe-nlp/gpt-neo-125m_instruction-tuned_sni"
    config.model_family = "gpt"
    config.data_dir = os.environ["MMLU_DATA_DIR"]
    config.predict_batch_size = 4
    config.max_input_length = 1024
    config.max_output_length = 128

    model = AutoModelForCausalLM.from_pretrained(config.model)
    model = model.cuda()

    print(eval_mmlu(config, model))
