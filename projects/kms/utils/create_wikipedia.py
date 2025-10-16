import json

from datasets import Dataset, load_dataset

for top_N in [5, 20, 100]:
    print(top_N)
    dataset = load_dataset("sordonia/my-wiki-latex_mmlu_from_valid_all", split="train")
    top_dataset = []

    for topic in set(dataset["subject"]):
        print(topic)
        mmlu = load_dataset("lighteval/mmlu", topic, split="test")
        filter_dataset = dataset.filter(lambda x: x["subject"] == topic, num_proc=16)
        filter_dataset = filter_dataset.sort("score", reverse=True)

        subject_text = ""
        for i in range(top_N):
            subject_text += (
                "New document: {}".format(filter_dataset[i]["title"])
                + "\n\n"
                + filter_dataset[i]["text"]
                + "\n\n"
            )

        questions = []
        choices = []
        gold_index = []
        for example in mmlu:
            questions.append(example["question"])
            choices.append(example["choices"])
            gold_index.append(example["answer"] + 1)

        top_dataset.append(
            {
                "document_id": topic,
                "text": subject_text,
                "questions": questions,
                "options": choices,
                "gold_label": gold_index,
                "split": "test",
            }
        )

    Dataset.from_list(top_dataset).push_to_hub(f"sordonia/wiki_top_{top_N}_sanitized")
