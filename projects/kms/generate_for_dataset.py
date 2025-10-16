import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

import click
import tenacity
import torch
import tqdm
import vllm
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from tqdm import tqdm as ttqdm
from tqdm.asyncio import tqdm as tqdm_async

from mttl.arguments import Args
from mttl.models.library.dataset_library import DatasetLibrary
from projects.kms.utils.dataset_augmenter import DatasetAugmenter


@dataclass
class AugmentArgs(Args):
    model: str = None
    dataset: str = None
    dataset_type: str = "narrativeqa"
    dataset_task: str = None
    block_size: int = 2048
    max_continuation_length: int = 768
    num_generations: int = 16
    generation_top_p: float = 0.8
    top_docs_per_subject: int = 50
    use_prompts: str = "summary,qa"
    output_path: str = "/tmp/"
    push_to_hub: str = None
    model_type: str = "vllm"
    do_filtering: bool = False


def main(args):
    import os

    from huggingface_hub import login

    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])

    augmenter = DatasetAugmenter(
        args.model,
        args.block_size,
        args.max_continuation_length,
        args.num_generations,
        args.generation_top_p,
        model_type=args.model_type,
        do_filtering=args.do_filtering,
    )
    for task in args.use_prompts.split(","):
        augmenter.add_task(task)

    concat_dataset = []
    if args.dataset_type == "wiki":
        dataset = load_dataset(dataset, split="train")

        for subject in tqdm.tqdm(
            dataset.unique("subject"), desc=f"Generating data for subjects"
        ):
            docs_per_subject = dataset.filter(
                lambda x: x["subject"] == subject, num_proc=16
            )
            docs_per_subject = docs_per_subject.sort("score", reverse=True).select(
                range(args.top_docs_per_subject)
            )

            # do augmentation
            output_dataset = augmenter.augment(docs_per_subject)

            # add subject column
            output_dataset = output_dataset.map(
                lambda x: {"subject": subject}, num_proc=16
            )

            # concat all datasets
            concat_dataset.extend(output_dataset.to_list())
            os.makedirs(args.output_path, exist_ok=True)

            d = DatasetDict(
                {
                    "train": Dataset.from_list(
                        concat_dataset, info=DatasetInfo(description=json.dumps(args))
                    )
                }
            )
            d.save_to_disk(args.output_path)
    elif args.dataset_type in ["narrativeqa", "quality", "wiki_top_20"]:
        # process only selected document ids
        import glob

        from datasets import disable_caching

        # disable caching
        disable_caching()
        dataset = load_dataset(f"sordonia/{args.dataset_type}_sanitized", split="train")
        dataset = DatasetLibrary.pull_dataset(args.dataset)["train"]

        if args.dataset_task is not None:
            if type(args.dataset_task) == tuple:
                document_ids = list(map(str, args.dataset_task))
            else:
                document_ids = args.dataset_task.split(",")

            dataset = dataset.filter(
                lambda x: str(x["document_id"]) in set(document_ids), num_proc=16
            )

        concat_dataset = augmenter.augment(dataset, carry_columns="document_id")
        os.makedirs(args.output_path, exist_ok=True)

        d = DatasetDict(
            {
                "train": Dataset.from_list(
                    concat_dataset,
                    info=DatasetInfo(description=args.to_json()),
                )
            }
        )
        d.save_to_disk(args.output_path)

    if args.push_to_hub is not None:
        d.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    args = AugmentArgs.parse()
    main(args)
