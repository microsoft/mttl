from mttl.models.library.dataset_library import DatasetLibrary
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
dataset = DatasetLibrary.pull_dataset_with_retry(
    "mlabonne/orca-agentinstruct-1M-v1-cleaned"
)["train"]


def get_source_target(example):
    message = example["messages"]
    output = tokenizer.apply_chat_template(message, tokenize=False)
    fields = output.split("<|assistant|>")
    source = fields[0] + "<|assistant|>"
    target = fields[1]
    example["source"] = source
    example["target"] = target
    example["task_name"] = example["split"]
    return example


dataset = dataset.map(
    get_source_target, remove_columns=["messages", "split"], num_proc=16
)

dataset.push_to_hub("zhan1993/orca_sqs_dataset_new")
