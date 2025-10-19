import json
import os
from collections.abc import Iterable

from transformers import AutoTokenizer, LlamaTokenizer

from mttl.logging import logger


def maybe_filter_hf_dataset_by_task(
    dataset,
    task_field,
    task_names: str = None,
    n_proc=16,
    should_split_on_split_column=True,
):
    """Filter a HuggingFace dataset by task names."""

    # get the tasks
    all_tasks = set()
    if "train" in dataset:
        all_tasks = all_tasks.union(set(dataset["train"][task_field]))
    if "validation" in dataset:
        all_tasks = all_tasks.union(set(dataset["validation"][task_field]))
    if "test" in dataset:
        all_tasks = all_tasks.union(set(dataset["test"][task_field]))

    if task_names:
        if isinstance(task_names, str):
            task_names = sorted(task_names.split(","))
        elif isinstance(task_names, Iterable):
            task_names = sorted(map(str, task_names))
        else:
            task_names = [str(task_names)]

        if not set(task_names).issubset(all_tasks):
            raise ValueError(
                "task_names must be a subset of the available tasks. Got {} and {}".format(
                    task_names, all_tasks
                )
            )

    train_dataset, dev_dataset, test_dataset = None, None, None
    if "train" in dataset:
        train_dataset = dataset["train"]
    if "validation" in dataset:
        dev_dataset = dataset["validation"]
    if "test" in dataset:
        test_dataset = dataset["test"]

    if (
        dev_dataset is None
        and test_dataset is None
        and "split" in train_dataset.features
        and should_split_on_split_column
    ):
        logger.info("Splitting train dataset on 'split' column.")
        train_dataset, dev_dataset, test_dataset = split_on_split_column(
            train_dataset, num_proc=n_proc
        )
    if task_names is not None:
        train_dataset = train_dataset.filter(
            lambda x: str(x[task_field]) in task_names,
            num_proc=n_proc,
            desc="Filtering task names",
        )
        if dev_dataset:
            dev_dataset = dev_dataset.filter(
                lambda x: str(x[task_field]) in task_names,
                num_proc=n_proc,
                desc="Filtering task names",
            )
        if test_dataset:
            test_dataset = test_dataset.filter(
                lambda x: str(x[task_field]) in task_names,
                num_proc=n_proc,
                desc="Filtering task names",
            )

    if task_names is None:
        task_names = list(all_tasks)

    task_to_id = {task: i for i, task in enumerate(task_names)}
    return task_names, task_to_id, train_dataset, dev_dataset, test_dataset


def split_on_split_column(dataset, num_proc=16):
    if "split" not in dataset.features:
        raise ValueError("Dataset does not have the required 'split' column!")

    train_dataset = dataset.filter(
        lambda x: x["split"] == "train",
        num_proc=num_proc,
        desc="Creating train set",
    )
    dev_dataset = dataset.filter(
        lambda x: x["split"] in ["dev", "validation", "valid"],
        num_proc=num_proc,
        desc="Creating valid set",
    )
    test_dataset = dataset.filter(
        lambda x: x["split"] in ["test"],
        num_proc=num_proc,
        desc="Creating test set",
    )
    return train_dataset, dev_dataset, test_dataset


def tokenizer_merges_space(tokenizer):
    test1 = "this"
    test2 = " this"

    return len(tokenizer(test1)["input_ids"]) == len(tokenizer(test2)["input_ids"])


def tokenizer_enforces_eos(tokenizer):
    test = "this is a long text seq that should be truncated"

    # copy tokenizer with add_eos parameter set to True
    old_add_eos = None

    if hasattr(tokenizer, "add_eos_token"):
        old_add_eos = tokenizer.add_eos_token
        tokenizer.add_eos_token = True

    token_ids = tokenizer(test, truncation=True, max_length=3)
    enforce_eos = token_ids["input_ids"][-1] == tokenizer.eos_token_id

    if old_add_eos is not None:
        tokenizer.add_eos_token = old_add_eos

    return enforce_eos


def get_tokenizer(config, for_generation=False):
    return get_tokenizer_with_args(
        config.model,
        config.model_family,
        config.padding_side,
        config.truncation_side,
        for_generation,
    )


def get_tokenizer_with_args(
    model_name,
    model_family="gpt",
    padding_side="right",
    truncation_side="right",
    for_generation=False,
):
    if "llama-2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
    else:
        if "phi-2" == model_name:
            # local phi-2 version. use `microsoft/phi-2 for the official hf version`
            model_name = os.environ["PHI_PATH"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.model_max_length = int(1e9)

    tokenizer.padding_side = padding_side
    logger.info("Padding side is {}".format(tokenizer.padding_side))

    tokenizer.truncation_side = truncation_side
    logger.info("Truncation side is {}".format(tokenizer.truncation_side))

    if model_family == "gpt":
        if for_generation:
            if padding_side == "right":
                logger.info("Padding side is 'right', but we are in generation mode!")

            logger.info(
                "for_generation is True, setting padding_side for tokenizer to 'left'."
            )
            tokenizer.padding_side = "left"

        # do not add eos token, we will add it accordingly *if* needed.
        tokenizer.add_eos_token = False

    if tokenizer.pad_token_id is None:
        logger.info(
            "Setting pad_token_id to eos_token_id, given that pad_token_id was not detected."
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.mttl_merges_space = tokenizer_merges_space(tokenizer)
    tokenizer.mttl_enforces_eos = tokenizer_enforces_eos(tokenizer)
    return tokenizer


def apply_custom_split_file(dataset, split_file):
    assert split_file.endswith(".json"), "split_file must be a json file"
    split_file = json.load(open(split_file, "r"))
    assert set(split_file.keys()) == {"train", "dev", "test"}

    doc_to_split = {
        doc: split for split in ["train", "dev", "test"] for doc in split_file[split]
    }
    all_docs = set(split_file["train"] + split_file["dev"] + split_file["test"])
    dataset = dataset.filter(lambda x: x["document_id"] in all_docs)

    def update_split(item):
        item["split"] = doc_to_split[item["document_id"]]
        return item

    # Update the Split column
    return dataset.map(update_split)
