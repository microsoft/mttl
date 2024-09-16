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
from dataset_augmenter import DatasetAugmenter
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from openai import AsyncAzureOpenAI
from tqdm import tqdm as ttqdm
from tqdm.asyncio import tqdm as tqdm_async


@click.command()
@click.option("--model", type=str, default="microsoft/Phi-3-medium-4k-instruct")
@click.option(
    "--dataset", type=str, default="sordonia/my-wiki-latex_mmlu_from_valid_all"
)
@click.option("--block_size", type=int, default=2048)
@click.option("--max_continuation_length", type=int, default=768)
@click.option("--num_generations", type=int, default=16)
@click.option("--generation_top_p", type=float, default=0.8)
@click.option("--top_docs_per_subject", type=int, default=50)
@click.option("--use_prompts", type=str, default="summary")
@click.option("--output_path", type=str, default="wiki_summaries")
@click.option("--push_to_hub", type=str, default=None)
def main(
    model,
    dataset,
    block_size,
    max_continuation_length,
    num_generations,
    generation_top_p,
    top_docs_per_subject,
    use_prompts,
    output_path,
    push_to_hub,
):
    args = {
        "model": model,
        "dataset": dataset,
        "block_size": block_size,
        "max_continuation_length": max_continuation_length,
        "num_generations": num_generations,
        "generation_top_p": generation_top_p,
        "top_docs_per_subject": top_docs_per_subject,
        "use_prompts": use_prompts,
        "output_path": output_path,
    }

    dataset = load_dataset("sordonia/my-wiki-latex_mmlu_from_valid_all", split="train")
    augmenter = DatasetAugmenter(
        model,
        block_size,
        max_continuation_length,
        num_generations,
        generation_top_p,
    )
    for task in use_prompts.split(","):
        augmenter.add_task(task)

    all_sections = []
    for subject in tqdm.tqdm(
        dataset.unique("subject"), desc=f"Generating data for subjects"
    ):
        docs_per_subject = dataset.filter(
            lambda x: x["subject"] == subject, num_proc=16
        )
        docs_per_subject = docs_per_subject.sort("score", reverse=True).select(
            range(top_docs_per_subject)
        )

        # do augmentation
        output_dataset = augmenter.augment(docs_per_subject)

        # add subject column
        output_dataset = output_dataset.map(lambda x: {"subject": subject}, num_proc=16)

        os.makedirs(output_path, exist_ok=True)

        d = DatasetDict(
            {
                "train": Dataset.from_list(
                    all_sections, info=DatasetInfo(description=json.dumps(args))
                )
            }
        )
        d.save_to_disk(output_path)

    if push_to_hub is not None:
        d.push_to_hub(push_to_hub)


if __name__ == "__main__":
    main()
