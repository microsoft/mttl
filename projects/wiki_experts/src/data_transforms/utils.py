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
    create_split=False,
    aug_few_shot=-1,
    cutoff=0,
    seed=42,
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
    rng = np.random.RandomState(seed)

    if create_split:
        pd = dataset.to_pandas()
        subjects = pd["subject"].unique()

        ds, tids = [], set()
        for sub in subjects:
            dataset_subject = dataset.filter(lambda x: x["subject"] == sub, num_proc=16)

            if cutoff > 0:
                dataset_subject = dataset_subject.shuffle(seed).select(
                    range(min(len(dataset_subject), cutoff))
                )

            # split 0.95 of ids for train, 0.05 for valid
            ids = list(set(dataset_subject["id"]))
            rng.shuffle(ids)
            stids = ids[: int(len(ids) * 0.95)]
            tids.update(stids)

            print(
                f"Subject {sub} has {len(ids)} examples, {len(stids)} for train, {len(ids) - len(stids)} for valid."
            )
            ds.append(dataset_subject)

    # creates a split column for each task (subject)
    def create_split_column(example):
        return {"split": "train"} if example["id"] in tids else {"split": "valid"}

    dataset = concatenate_datasets(ds)
    dataset = dataset.map(create_split_column, num_proc=16)

    def rename_columns(example):
        example["source"] = example["instruction"]
        example["target"] = example["response"]
        example["task_name"] = example["subject"]
        example["task_source"] = "oai-mc-mmlu"
        example["split"] = example["split"]
        return example

    dataset = dataset.map(
        rename_columns,
        num_proc=16,
        remove_columns=["instruction", "response", "subject"],
    )

    if aug_few_shot > 0:
        # augment the dataset few-shot
        from mttl.datamodule.mt_seq_to_seq_module import (
            augment_few_shot,
        )
        from datasets import concatenate_datasets

        aug_dataset = augment_few_shot(dataset, aug_few_shot)
        dataset = concatenate_datasets([aug_dataset, dataset]).shuffle(seed)

    dataset.push_to_hub(hf_destination, token=hf_token)
    dataset.to_json(
        f"{hf_destination.replace('/', '_')}.json", orient="records", lines=True
    )

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
