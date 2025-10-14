import click
from datasets import load_dataset

from mttl.models.library.dataset_library import DatasetLibrary


@click.command()
@click.option("--hf_dataset", type=str, required=True)
@click.option("--blob_storage", type=str, required=True)
def main(hf_dataset, blob_storage):
    if not hf_dataset:
        raise ValueError("Please provide a dataset name!")

    data = load_dataset(hf_dataset)

    print("Pushing to: ", blob_storage)
    DatasetLibrary.push_dataset(data, blob_storage)


if __name__ == "__main__":
    main()
