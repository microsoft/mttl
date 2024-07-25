import json
import os

import click
from datasets import load_dataset
from transformers import AutoTokenizer


@click.command()
@click.option("--input_jsonl", help="Path to the input jsonl file")
@click.option("--model", help="Model name or path to the model")
@click.option(
    "--output_dataset",
    help="Path to the output hf dataset. Same as input file but with no extension if not provided.",
    default=None,
)
def main(input_jsonl, model, output_dataset):
    if output_dataset is None:
        output_dataset, _ = os.path.splitext(input_jsonl)

    num_proc = os.environ.get("MTTL_NUM_PROC_DATASETS", 16)

    tokenizer = AutoTokenizer.from_pretrained(model)

    def apply_chat_template(example):
        return tokenizer.apply_chat_template(
            example,
            tokenize=False,
            add_generation_prompt=False,
        )

    def chat_progression(examples):
        """Split chat into individual chat turns. For each turn, the source is
        the chat up to that point and the target is the assistant's message."""
        sources = []
        targets = []
        task_names = []
        num_rounds = []
        for messages, metadata in zip(examples["messages"], examples["metadata"]):
            messages = json.loads(messages)
            task_name = json.loads(metadata or "{}").get("task_name", "unknown")
            chat_progression = []
            rounds = 1
            for message in messages:
                if message["role"] != "assistant":
                    chat_progression.append(message)
                else:
                    sources.append(apply_chat_template(list(chat_progression)))
                    targets.append(apply_chat_template([dict(message)]))
                    task_names.append(task_name)
                    num_rounds.append(rounds)
                    chat_progression.append(message)
                    rounds += 1
        return {
            "source": sources,
            "target": targets,
            "task_name": task_names,
            "round": num_rounds,
        }

    dataset = load_dataset("json", data_files=input_jsonl)

    dataset = dataset.map(
        chat_progression,
        batched=True,  # allows to return more examples than the input
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
    )

    dataset.save_to_disk(output_dataset)


if __name__ == "__main__":
    main()
