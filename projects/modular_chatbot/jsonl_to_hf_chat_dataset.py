import json
import os

import click
from datasets import load_dataset
from transformers import AutoTokenizer


def normalize_tag(tag):
    tag = tag.lower()
    tag = tag.replace(" ", "_")
    return tag


def custom_transform(messages):
    text = ""
    for idx, message in enumerate(messages):
        assert message["role"] in ["system", "user", "assistant"]

        text += f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>\n"
    return text


@click.command()
@click.option("--input_jsonl", help="Path to the input jsonl file")
@click.option("--model", help="Model name or path to the model")
@click.option(
    "--output_dataset",
    help="Path to the output hf dataset. Same as input file but with no extension if not provided.",
    default=None,
)
def main(input_jsonl, model=None, output_dataset=None):
    if output_dataset is None:
        output_dataset, _ = os.path.splitext(input_jsonl)

    num_proc = os.environ.get("MTTL_NUM_PROC_DATASETS", 16)

    if model is not None:
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
        last_tags = []
        last_tag = "[EMPTY]"

        for messages, metadata in zip(examples["messages"], examples["metadata"]):
            task_name = json.loads(metadata or "{}").get("task_name", "unknown")
            rounds = 1
            chat_progression = []
            for message in messages:
                if message["role"] != "assistant":
                    chat_progression.append(message)
                    # if this message has been tagged, then update the last tag and consider that tag for the next assistant message
                    if message["role"] == "user" and "tag" in message:
                        last_tag = normalize_tag(message["tag"])
                    else:
                        last_tag = "[EMPTY]"
                else:
                    if len(chat_progression) == 1:
                        continue

                    func = (
                        apply_chat_template if model is not None else custom_transform
                    )
                    sources.append(func(list(chat_progression)))
                    targets.append(func([message]))
                    task_names.append(task_name)
                    num_rounds.append(rounds)
                    chat_progression.append(message)
                    last_tags.append(last_tag)
                    rounds += 1

        return {
            "source": sources,
            "target": targets,
            "task_name": task_names,
            "round": num_rounds,
            "tag": last_tags,
        }

    dataset = load_dataset("json", data_files=input_jsonl)

    dataset = dataset.map(
        chat_progression,
        batched=True,  # allows to return more examples than the input
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )

    dataset.save_to_disk(output_dataset)


if __name__ == "__main__":
    main()
