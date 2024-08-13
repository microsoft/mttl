import click

from mttl.dataloader.flan_utils import download_flan
from mttl.models.library.expert_library import DatasetLibrary


@click.command()
@click.argument("task")
@click.option(
    "--split",
    type=str,
    default="train",
    help="Split to download",
)
@click.option(
    "--download_size",
    type=int,
    default="-1",
    help=(
        "Download size from Hugging Face (before any processing). "
        "Values <= 0 means download the whole dataset"
    ),
)
@click.option(
    "--cutoff",
    type=int,
    default=10_000,
    help="Max number of examples per task. Zero means no cutoff.",
)
@click.option(
    "--dataset_library_id",
    type=str,
    default=None,
    help="Repository ID where to push the processed dataset.",
)
@click.option(
    "--verbose/--no-verbose",
    default=True,
    help="Print splits information for each task.",
)
def main(task, split, download_size, cutoff, dataset_library_id, verbose):
    if task == "flan":
        concatenated_datasets = download_flan(
            split=split,
            download_size=download_size,
            cutoff=cutoff,
            verbose=verbose,
        )
    else:
        raise ValueError("Unknown task")

    if dataset_library_id is not None:
        print("Pushing dataset to ", dataset_library_id)
        DatasetLibrary.push_dataset(concatenated_datasets, dataset_library_id)
    print("Done!")


if __name__ == "__main__":
    main()
