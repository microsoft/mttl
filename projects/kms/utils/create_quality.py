import json

from datasets import Dataset, DatasetDict

from mttl.models.library import dataset_library

push_to = None

data = {}
with open("./quality_data/QuALITY.v1.0.1.htmlstripped.dev", "r") as f:
    data["valid"] = [json.loads(l) for l in f]

with open("./quality_data/QuALITY.v1.0.1.htmlstripped.test", "r") as f:
    data["test"] = [json.loads(l) for l in f]

with open("./quality_data/QuALITY.v1.0.1.htmlstripped.train", "r") as f:
    data["train"] = [json.loads(l) for l in f]

dataset = []
for split in ["train", "valid", "test"]:
    for line in data[split]:
        dataset.append(
            {
                "split": split,
                "document_id": line["article_id"],
                "questions": [
                    line["questions"][i]["question"]
                    for i in range(len(line["questions"]))
                ],
                "options": [
                    line["questions"][i]["options"]
                    for i in range(len(line["questions"]))
                ],
                "gold_label": [
                    line["questions"][i].get("gold_label", -1)
                    for i in range(len(line["questions"]))
                ],
                "difficult": [
                    line["questions"][i]["difficult"]
                    for i in range(len(line["questions"]))
                ],
                "text": line["article"],
            }
        )

# for every entry in dataset if document_id match, keep only one and extend all fields that are lists
# this is done to avoid duplicates in the dataset
new_dataset = []
for entry in dataset:
    found = False
    for new_entry in new_dataset:
        if new_entry["document_id"] == entry["document_id"]:
            found = True
            for key in entry:
                if isinstance(entry[key], list):
                    new_entry[key].extend(entry[key])
    if not found:
        new_dataset.append(entry)

dataset_dict = DatasetDict({"train": Dataset.from_list(new_dataset)})
dataset_library.DatasetLibrary.push_dataset(dataset_dict, push_to)
