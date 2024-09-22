import argparse
import os

import tqdm
from datasets import Dataset, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_id", type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset("deepmind/narrativeqa")
    document_ids = {}
    for split in ["train", "validation", "test"]:
        for example in dataset[split]:
            document = example["document"]
            if document["id"] not in document_ids:
                document_ids[document["id"]] = {
                    "text": document["text"],
                    "questions": [],
                    "answers": [],
                    "document_id": document["id"],
                    "split": split,
                }
            else:
                document_ids[document["id"]]["questions"] += [
                    example["question"]["text"]
                ]
                document_ids[document["id"]]["answers"] += [
                    [a["text"] for a in example["answers"]]
                ]

    dataset = list(document_ids.values())
    Dataset.from_list(dataset).push_to_hub(args.hf_id)


if __name__ == "__main__":
    main()
