import json

from datasets import Dataset

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

dataset = Dataset.from_list(dataset).push_to_hub("sordonia/quality_sanitized")
