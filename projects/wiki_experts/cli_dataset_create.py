import itertools
import json
from datasets import load_dataset, concatenate_datasets, Dataset
from promptsource import templates
import tqdm
from mttl.datamodule.t0_data_module import apply_template
from mttl.dataloader import t0_dataset_readers
from mttl.datamodule import t0_data_module
import os


def download_t0(cutoff=-1, per_task=True):
    dataset_folder = "t0_task"
    dataset = t0_dataset_readers.T0MixtureReader(
        t0_dataset_readers.T0DatasetConfig(
            "t0", seed=42, use_t0_templates_as_tasks=False
        )
    )

    # filter some examples from the dataset
    datasets = dataset.read_orig_dataset("train")
    all_templates = templates.TemplateCollection()

    task_dict = {}
    for task_dataset in datasets:
        template = all_templates.get_dataset(
            task_dataset.dataset_name, task_dataset.subset_name
        )[task_dataset.template_name]
        task_name = task_dataset.dataset_name + (
            ("/" + task_dataset.subset_name) if task_dataset.subset_name else ""
        )
        if task_name not in task_dict:
            task_dict[task_name] = []

        def map_func(example):
            try:
                source, target = apply_template(template, example, hash_friendly=True)
            except:
                source = "<NO INPUT>"
                target = "<NO LABEL>"
            return {
                "source": source,
                "target": target,
                "task_name": task_name,
                "template_type": task_dataset.template_name,
                "task_source": "T0",
            }

        column_names = [
            column
            for column in task_dataset.column_names
            if column
            not in ["source", "target", "task_name", "template_type", "task_source"]
        ]
        map_dataset = task_dataset.map(
            map_func, remove_columns=column_names, num_proc=32
        )
        map_dataset = map_dataset.filter(
            lambda x: x["source"] != "<NO INPUT>" and x["target"] != "<NO LABEL>"
        )
        task_dict[task_name].append(map_dataset)

    if per_task:
        for task_name, task_dataset in task_dict.items():
            print("Dumping task", task_name)
            task_dataset = concatenate_datasets(task_dataset)
            # if the dataset is too large, we randomly sample 5000 examples for the training
            if cutoff > 0:
                if len(task_dataset) > cutoff:
                    task_dataset = task_dataset.shuffle()
                    task_dataset = task_dataset.select(range(cutoff))
            # save it into the task file
            task_name = task_name.replace("/", "_")
            task_dataset.to_json(os.path.join(dataset_folder, task_name + ".json"))
    else:
        assert cutoff > 0
        all_dataset = concatenate_datasets(
            list(itertools.chain(*list(task_dict.values())))
        )
        all_dataset = all_dataset.shuffle().select(range(cutoff))
        task_names = task_dict.keys()

        for task_name in task_names:
            task_dataset = all_dataset.filter(lambda x: x["task_name"] == task_name)
            task_name = task_name.replace("/", "_")
            task_dataset.to_json(os.path.join(dataset_folder, task_name + ".json"))


def download_flan(cutoff=10_000, filter_zs=False):
    dataset_folder = "flan_task"
    dataset = load_dataset("chiayewken/flan-v2", split="train")

    # filter some examples from the dataset
    if filter_zs:
        dataset = dataset.filter(
            lambda example: example["template_type"] == "zs_noopt", num_proc=32
        )

    # group the dataset using the task_name
    task_names = dataset.unique("task_name")
    print("Num Tasks: ", len(task_names))

    task_dict = {task_name: [] for task_name in task_names}
    for idx, example in tqdm.tqdm(enumerate(dataset)):
        task_dict[example["task_name"]].append(example)

    for task_name, task_dataset in task_dict.items():
        if len(task_dataset) == 0:
            continue
        task_dict[task_name] = Dataset.from_list(task_dataset)

    for task_name in task_names:
        print("Processing task: ", task_name)

        task_dataset = task_dict[task_name]
        # if the dataset is too large, we randomly sample 5000 examples for the training
        task_dataset = task_dataset.shuffle()

        if len(task_dataset) > cutoff:
            task_dataset = task_dataset.select(range(cutoff))

        len_dataset_ = len(task_dataset)

        num_train = int(len_dataset_ * 0.8)
        num_test = int(len_dataset_ * 0.1)

        def assign_split(example, idx):
            if idx < num_train:
                return {"split": "train"}
            elif num_train <= idx < num_train + num_test:
                return {"split": "test"}
            else:
                return {"split": "validation"}

        task_dataset = task_dataset.map(assign_split, with_indices=True)

        # save it into the task file
        task_name = task_name.replace("/", "_")
        task_dataset.to_json(os.path.join(dataset_folder, task_name + ".json"))

        print("Dumping task", task_name)
        print("# Train", len(task_dataset.filter(lambda x: x["split"] == "train")))
        print("# Test", len(task_dataset.filter(lambda x: x["split"] == "test")))
        print("# Valid", len(task_dataset.filter(lambda x: x["split"] == "validation")))


def create_data(dataset_folder, hf_destination, flat=True):
    import glob
    from datasets import DatasetDict
    import huggingface_hub

    hf_token = os.environ.get("HF_TOKEN")
    huggingface_hub.login(token=hf_token)

    files = glob.glob(os.path.join(dataset_folder, "*.json"))
    dataset_dict = DatasetDict()

    for file in files:
        dataset = load_dataset("json", data_files=file)["train"]

        def clean_task(x):
            if "task_name" not in x:
                return x

            x["task_name"] = (
                x["task_name"]
                .replace(":", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
            )
            return x

        dataset = dataset.map(lambda x: clean_task(x))
        task_name = dataset["task_name"][0]
        print(f"Loading {task_name}")
        dataset_dict[task_name] = dataset

    if flat:
        datasets = list(dataset_dict.values())
        concatenate_datasets(datasets).push_to_hub(hf_destination, token=hf_token)
    else:
        dataset_dict.push_to_hub(hf_destination, token=hf_token)


if __name__ == "__main__":
    task = "flan"
    if task == "flan":
        download_flan(cutoff=10_000)
        create_data("flan_task", "sordonia/flan-10k-flat", flat=True)
    elif task == "t0":
        download_t0(cutoff=10_000)
        create_data("t0_task", "sordonia/t0-10k-flat", flat=True)
