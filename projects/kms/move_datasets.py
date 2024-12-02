import click
from datasets import load_dataset

from mttl.models.library.dataset_library import DatasetLibrary


@click.command()
@click.option("--hf_dataset", type=str, required=True)
@click.option("--blob_storage", type=str, default="mttldata")
def main(hf_dataset, blob_storage):
    if not hf_dataset:
        raise ValueError("Please provide a dataset name!")

    # blob storage doesn't like "_" in the name
    name = hf_dataset.split("/")[-1].replace("_", "-")

    wiki = load_dataset("sordonia/wiki_top_20_sanitized_rag")

    print("Pushing to: ", f"az://{blob_storage}/{name}")
    DatasetLibrary.push_dataset(wiki, f"az://{blob_storage}/{name}")


if __name__ == "__main__":
    main()
