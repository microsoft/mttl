import argparse
import os

import tqdm
from datasets import Dataset, load_dataset


def normalize_narrativeqa(text):
    # narrativeqa related text normalization
    if "*** START OF THIS PROJECT" in text:
        text = text.split("*** START OF THIS PROJECT")[1]
    if "***START OF THE PROJECT" in text:
        text = text.split("***START OF THE PROJECT")[1]
    if "*** END OF THIS PROJECT" in text:
        text = text.split("*** END OF THIS PROJECT")[0]
    if "***END OF THE PROJECT" in text:
        text = text.split("***END OF THE PROJECT")[0]
    text = text.split("<pre>")[-1]
    text = text.split("</pre>")[0]
    text = text.replace("<b>", "").replace("</b>", "")
    text = text.replace("[Illustration]", "")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_id", type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset("deepmind/narrativeqa")
    document_ids = {}
    for split in ["train", "validation", "test"]:
        for example in tqdm.tqdm(dataset[split]):
            document = example["document"]
            if document["id"] not in document_ids:
                document_ids[document["id"]] = {
                    "text": normalize_narrativeqa(document["text"]),
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
