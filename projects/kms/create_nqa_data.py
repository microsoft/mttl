import argparse
import os

import tqdm
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset("deepmind/narrativeqa")
    document_ids = {}
    for split in ["train", "validation", "test"]:
        for document in dataset[split]:
            document = document["document"]
            if document["id"] not in document_ids:
                document_ids[document["id"]] = document["text"]

    os.makedirs(args.output_path, exist_ok=True)

    for document_id, text in tqdm.tqdm(
        document_ids.items(),
        total=len(document_ids),
        desc="Writing documents to disk...",
    ):
        with open(f"{args.output_path}/{document_id}.txt", "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
