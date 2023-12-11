import json
import os
import re
from datasets import load_dataset
import numpy as np
from mttl.utils import retry_with_exponential_backoff


INVALID_RESPONSE = object()


def dump_jsonl_dataset(dataset, filename):
    """Writes a jsonl dataset into a file."""
    with open(filename, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry))
            f.write("\n")


def count_repeating_sentences(text):
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    unique_sentences = set(sentences)
    return len(sentences) - len(unique_sentences)


def reject_output(output, finish_reason):
    if isinstance(output, tuple):
        output = output[0]

    # common error here is that instruxtions starts with BEGININPUT + some nonsense, let evoid it
    # usually, if the instruction is too long, its a bad one
    # or if it stopped due to max tokens
    return output is INVALID_RESPONSE or finish_reason != "stop"


def read_jsonl_dataset(filename):
    """Read a jsonl dataset into a list of dicts."""
    with open(filename, "r") as f:
        dataset = [json.loads(s) for s in f.readlines()]
    return dataset


def upload_to_hf_(
    dataset_path,
    hf_destination=None,
    configuration=None,
    flat=False,
    create_split=False,
):
    import pandas as pd
    import huggingface_hub

    hf_token = os.environ.get("HF_TOKEN")
    huggingface_hub.login(token=hf_token)

    if hf_destination is None:
        dts_name = dataset_path.split("/")[-1].replace(".jsonl", "")
        hf_user = huggingface_hub.whoami()["name"]
        hf_destination = f"{hf_user}/{dts_name}"

    dataset = load_dataset("json", data_files=dataset_path)["train"]

    if create_split:
        pd = dataset.to_pandas()
        subjects = pd["subject"].unique()

        train_ids = []
        valid_ids = []
        test_ids = []

        for sub in subjects:
            dts_subject = dataset.filter(lambda x: x["subject"] == sub, num_proc=16)
            context_ids = list(set(dts_subject["id"]))
            train_ids = train_ids + context_ids[: int(len(context_ids) * 0.95)]
            valid_ids = valid_ids + context_ids[int(len(context_ids) * 0.95) :]
            print(sub, len(context_ids))
            print("train", len(train_ids))
            print("valid", len(valid_ids))

        # creates a split column for each task (subject)
        def create_split_column(example):
            return (
                {"split": "train"} if example["id"] in train_ids else {"split": "valid"}
            )

        dataset = dataset.map(create_split_column, num_proc=16)

    if flat:

        def rename_columns(example):
            example["source"] = example["instruction"]
            example["target"] = example["response"]
            example["task_name"] = example["subject"]
            example["task_source"] = "oai-mc-mmlu"
            example["split"] = example["split"]
            return example

        if configuration is not None and "multichoice" in configuration.model_setting:

            def normalize_response_and_shuffle_options(example):
                if example["response"][0] not in "ABCD":
                    example["response"] = "invalid"
                    return example

                example["response"] = example["response"][0]
                # Find all matches in the input string
                pattern = r"Question:\n(.*?)\nChoices:\n(.*?)\nAnswer:"
                question, choices = re.findall(
                    pattern, example["instruction"], re.DOTALL
                )[0]
                choices = re.findall(r"(\w)\. ([^\n]+)", choices)

                if len(choices) != 4:
                    example["response"] = "invalid"
                    return example

                # error in the labels
                labels = [c[0] for c in choices]
                if labels != ["A", "B", "C", "D"]:
                    example["response"] = "invalid"
                    return example

                # restrict the number of options to 4 and re-assign options
                np.random.shuffle(choices)
                labels, texts = zip(*choices)

                # case in which the new answer is not in ABCD
                new_answer = "ABCD"[labels.index(example["response"])]
                new_choices = "\n".join(
                    [f"{label}. {text}" for label, text in zip("ABCD", texts[:4])]
                )
                example[
                    "instruction"
                ] = f"Question:\n{question}\nChoices:\n{new_choices}\nAnswer:"
                example["response"] = new_answer
                return example

            dataset = (
                dataset.map(normalize_response_and_shuffle_options, num_proc=16)
                .filter(lambda x: x["response"] != "invalid", num_proc=16)
                .map(
                    rename_columns,
                    num_proc=16,
                    remove_columns=["instruction", "response", "subject"],
                )
            )

    dataset.push_to_hub(hf_destination, token=hf_token)

    if configuration is not None:
        from huggingface_hub import HfApi

        api = HfApi()
        setting_dict = configuration.__dict__

        with open("/tmp/readme.txt", "w") as f:
            for k, v in setting_dict.items():
                f.write(f"## {k}: {v}\n")

        @retry_with_exponential_backoff(
            errors=huggingface_hub.utils._errors.HfHubHTTPError
        )
        def upload():
            api.upload_file(
                path_or_fileobj="/tmp/readme.txt",
                path_in_repo="README.md",
                repo_id=hf_destination,
                repo_type="dataset",
                token=hf_token,
            )

        upload()
        os.remove("/tmp/readme.txt")
