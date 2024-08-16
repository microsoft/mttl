import json
import os

import click
from datasets import load_dataset
from transformers import AutoTokenizer


@click.command()
@click.option("--input_jsonl", help="Path to the input jsonl file")
@click.option("--output_jsonl", help="Model name or path to the model")
def main(input_jsonl, output_jsonl):
    num_proc = os.environ.get("MTTL_NUM_PROC_DATASETS", 16)

    def prettify(examples):
        examples_ = {key: value for key, value in examples.items()}
        examples_["messages"] = []
        examples_["metadata"] = []
        examples_["task_name"] = []
        for messages, metadata in zip(examples["messages"], examples["metadata"]):
            messages = json.loads(messages)
            task_name = json.loads(metadata or "{}").get("task_name", "unknown")
            examples_["messages"].append(messages)
            examples_["metadata"].append(metadata)
            examples_["task_name"].append(task_name)
        return examples_

    dataset = load_dataset("json", data_files=input_jsonl)
    dataset = dataset.map(
        prettify,
        batched=True,  # allows to return more examples than the input
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
    )
    dataset["train"].to_json(output_jsonl)


if __name__ == "__main__":
    main()
