import click
from datasets import Dataset, DatasetDict, load_dataset


@click.command()
@click.argument("dataset_path", type=str)
@click.option("--hf_destination", type=str, default=None)
def convert_summaries_to_dataset(dataset_path: str, hf_destination: str):
    # the directory is a bunch of json files, each file is a summary of a wikipedia article
    # each json file has a "text" field, which is the article
    # we want to create a dataset with the following fields:
    # - text: the article
    # - summary: the summary
    import json
    import os

    dataset = []
    for file in os.listdir(dataset_path):
        with open(os.path.join(dataset_path, file), "r") as f:
            data = json.load(f)
            # data is a list of entries, each entry is a dictionary with a "text" field and a "summaries" field
            for doc_id, entry in enumerate(data):
                # each data["summaries"] is a list of summaries
                # just store them in a list dumped as a json string
                subject = file.split(".")[0]
                dataset.append(
                    {
                        "text": entry["text"],
                        "summaries": entry["summaries"],
                        "subject": subject,
                    }
                )

    # save the dataset to a dataset
    dataset = DatasetDict({"train": Dataset.from_list(dataset)})
    if hf_destination:
        dataset.push_to_hub(hf_destination)
    dataset.save_to_disk(dataset_path + "_dataset")


if __name__ == "__main__":
    convert_summaries_to_dataset()
