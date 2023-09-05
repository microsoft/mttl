from datasets import Dataset, load_dataset
import json
import os


settings = {
    "all": -1,
    "1M": 1_000_000,
    "10M": 10_000_000,
}
split = "validation"
filename = f"documents_by_subject_{split}.json"

wikipedia = load_dataset("wikipedia", "20220301.en")

for setting, total_num_tokens in settings.items():
    with open(filename, "rt") as f:
        documents_by_subject = json.load(f)

    dataset = {
        "subject": [],
        "docno": [],
        "text": [],
        "score": [],
        "dfq": [],
        "title": [],
    }

    for subject, documents in documents_by_subject.items():
        num_tokens = 0

        for j, document in enumerate(documents):
            docno = document[0]
            text = wikipedia["train"][int(docno)]["text"]
            dataset["text"].append(text)
            dataset["subject"].append(subject)
            dataset["docno"].append(int(docno))
            dataset["dfq"].append(document[1]["dfq"])
            dataset["score"].append(document[1]["score"])
            dataset["title"].append(document[1]["title"])
            num_tokens += len(text.split())
            if num_tokens > total_num_tokens and total_num_tokens != -1:
                break

        print(f"=====================")
        print(f"Subject: {subject}")
        print(f"Number of documents: {len(documents)}")
        print(f"Number of tokens: {num_tokens}")
        print(f"Number of added documents: {j + 1}")

    dataset = Dataset.from_dict(dataset)
    dataset.push_to_hub(
        "sordonia/wiki_mmlu_{}{}".format(
            setting, f"_from_{split}" if split != "test" else ""
        ),
        token=os.environ.get("HF_TOKEN"),
    )
