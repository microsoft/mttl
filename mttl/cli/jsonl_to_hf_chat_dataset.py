import os
import json

import click
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@click.command()
@click.option('--input_jsonl', help='Path to the input jsonl file')
@click.option('--model', help='Model name or path to the model')
@click.option(
    '--output_dataset',
    help='Path to the output hf dataset. Same as input file but withouth the extension if not provided.',
    default=None,
)
def main(input_jsonl, model, output_dataset):
    if output_dataset is None:
        output_dataset, _ = os.path.splitext(input_jsonl)
    dataset = load_dataset("json", data_files=input_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    dataset = dataset.map(
        lambda x: {
            "formatted_chat": tokenizer.apply_chat_template(
                json.loads(x["messages"]),
                tokenize=False,
                add_generation_prompt=False,
            ),
            "task_name": json.loads(
                x["metadata"] if x["metadata"] else "{}"
            ).get("task_name", "unknown"),
        },
        num_proc=os.environ.get("MTTL_NUM_PROC_DATASETS", 16),
    )

    dataset.save_to_disk(output_dataset)


if __name__ == "__main__":
    main()

