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
from tqdm import tqdm as ttqdm
from tqdm.asyncio import tqdm as tqdm_async


@click.command()
@click.option("--model", type=str, default="microsoft/Phi-3-medium-4k-instruct")
@click.option(
    "--dataset", type=str, default="sordonia/my-wiki-latex_mmlu_from_valid_all"
)
@click.option("--dataset_type", type=str, default="wiki", help="wiki, narrativeqa")
@click.option(
    "--dataset_task",
    type=str,
    default=None,
    help="For wiki, the subject. For narrativeqa, the document_id.",
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
    dataset_type,
    dataset_task,
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
        "dataset_type": dataset_type,
        "dataset_task": dataset_task,
        "block_size": block_size,
        "max_continuation_length": max_continuation_length,
        "num_generations": num_generations,
        "generation_top_p": generation_top_p,
        "top_docs_per_subject": top_docs_per_subject,
        "use_prompts": use_prompts,
        "output_path": output_path,
    }

    augmenter = DatasetAugmenter(
        model,
        block_size,
        max_continuation_length,
        num_generations,
        generation_top_p,
    )
    for task in use_prompts.split(","):
        augmenter.add_task(task)

    concat_dataset = []
    if dataset_type == "wiki":
        dataset = load_dataset(dataset, split="train")

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
            output_dataset = output_dataset.map(
                lambda x: {"subject": subject}, num_proc=16
            )

            # concat all datasets
            concat_dataset.extend(output_dataset.to_list())
            os.makedirs(output_path, exist_ok=True)

            d = DatasetDict(
                {
                    "train": Dataset.from_list(
                        concat_dataset, info=DatasetInfo(description=json.dumps(args))
                    )
                }
            )
            d.save_to_disk(output_path)
    elif dataset_type == "narrativeqa":
        # process only selected document ids
        import glob

        # infer docids from txt files in the dataset dir, but strip the extension
        document_ids = [
            os.path.basename(f).split(".")[0] for f in glob.glob(f"{dataset}/*.txt")
        ]

        if dataset_task is not None:
            document_ids = dataset_task.split(",")

        texts = []
        for document_id in tqdm.tqdm(
            list(document_ids), desc=f"Generating data for documents"
        ):
            text = open(f"{dataset}/{document_id}.txt").read()

            # narrativeqa related text normalization
            text = text.split("*** END OF THIS PROJECT")[0]
            text = text.split("<pre>")[-1]
            text = text.split("</pre>")[0]
            text = text.replace("<b>", "").replace("</b>", "")
            texts.append({"text": text, "document_id": document_id})

        # do augmentation
        concat_dataset = augmenter.augment(Dataset.from_list(texts))
        os.makedirs(output_path, exist_ok=True)

        d = DatasetDict(
            {
                "train": Dataset.from_list(
                    concat_dataset,
                    info=DatasetInfo(description=json.dumps(args)),
                )
            }
        )
        d.save_to_disk(output_path)

    if push_to_hub is not None:
        d.push_to_hub(push_to_hub)


if __name__ == "__main__":
    main()
