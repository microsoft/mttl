import json
import os
import re
from datasets import load_dataset

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
    return (
        output is INVALID_RESPONSE
        or "BEGININPUT" in output
        or not output.strip()
        or finish_reason != "stop"
    )


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
    from datasets import DatasetDict
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
            dts_subject = dataset.filter(lambda x: x["subject"] == sub)
            context_ids = list(set(dts_subject["id"]))
            train_ids = train_ids + context_ids[: int(len(context_ids) * 0.9)]
            valid_ids = valid_ids + context_ids[int(len(context_ids) * 0.9) :]
            print(sub, len(context_ids))
            print("train", len(train_ids))
            print("valid", len(valid_ids))

        # creates a split column for each task (subject)
        def create_split_column(example):
            return (
                {"split": "train"} if example["id"] in train_ids else {"split": "valid"}
            )

        dataset = dataset.map(create_split_column)

    if not flat:
        pd = dataset.to_pandas()
        subjects = pd["subject"].unique()

        dts_per_subject = DatasetDict()
        for sub in subjects:
            dts_subject = dataset.filter(lambda x: x["subject"] == sub)
            dts_per_subject[sub] = dts_subject

        dts_per_subject.push_to_hub(hf_destination, token=hf_token)
    else:
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
