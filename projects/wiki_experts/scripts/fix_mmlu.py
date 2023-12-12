from datasets import load_dataset


def fix_punctuation(example):
    example["target"] = example["target"].strip()
    example["source"] = example["source"].strip()
    if example["target"][0] == "[" and example["target"][-1] == "]":
        example["target"] = example["target"][1:-1]
    if example["source"][-1] in [";", ",", ":"]:
        example["source"] = example["source"][:-1]
    if example["target"][-1] in [";", ",", ":"]:
        example["target"] = example["target"][:-1]
    if example["source"][-1] not in [".", "?", "!"]:
        example["source"] += "."
    if example["target"][-1] not in [".", "?", "!"]:
        example["target"] += "."
    return example


def remove_empty_targets(example):
    return len(example["target"].strip())


def remove_please(example):
    # Remove examples that start with "Please", as they are not answers
    return not example["target"].startswith("Please")


dataset = load_dataset("sordonia/mmlu-qa-flat")["train"]

print("Number of examples before removing empty targets:", len(dataset))
dataset = dataset.filter(remove_empty_targets)
print("Number of examples after removing empty targets:", len(dataset))

print("Number of examples before removing 'Please':", len(dataset))
dataset = dataset.filter(remove_please)
print("Number of examples after removing 'Please':", len(dataset))

print("Number of examples before fixing punctuation:", len(dataset))
dataset = dataset.map(fix_punctuation, num_proc=16)
print("Number of examples after fixing punctuation:", len(dataset))

dataset.push_to_hub("sordonia/mmlu-qa-flat")
