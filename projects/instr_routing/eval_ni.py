from mttl.datamodule.utils import get_tokenizer
from projects.instr_routing.finetune_llama import RoutingConfig
from projects.instr_routing.models.clm import CLM
from transformers import AutoModelForCausalLM
import os
import torch


def eval_ni(
    config,
    model,
    nshot=2,
    data_dir=None,
    eval_batches=-1,
):
    from mttl.evaluators import NIEvaluator

    ni_evaluator = NIEvaluator(
        config,
        data_dir=data_dir or config.data_dir,
        num_pos_examples=nshot
    )
    metrics = ni_evaluator.evaluate(model, metric_per_task=True, eval_batches=eval_batches)
    torch.cuda.empty_cache()
    return metrics["rougeL"]["all"]


if __name__ == "__main__":
    from huggingface_hub import login

    config = RoutingConfig.parse()

    login(token=os.environ["HF_TOKEN"])

    config = RoutingConfig.parse(extra_kwargs={"eval_superni": True})

    config.model = "meta-llama/Llama-2-7b-hf"
    config.load_in_8bit = True
    config.model_family = "gpt"
    config.data_dir = os.environ["NI_DATA_DIR"]
    config.predict_batch_size = 2
    config.max_input_length = 4096
    config.max_output_length = 128

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        load_in_8bit=config.load_in_8bit,
        device_map="auto"
    )

    print(eval_ni(config, model, nshot=2, eval_batches=50))
