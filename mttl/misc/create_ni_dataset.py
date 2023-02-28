import copy
import os
import click
import pytorch_lightning as pl
import json
import random
import rich


def load_instances(task, data_path, num_examples=100, is_test=False):
    task_path = os.path.join(data_path, "tasks", task + ".json")

    with open(task_path, encoding="utf-8") as task_f:
        s = task_f.read()

    task_data = json.loads(s)
    all_instances = task_data.pop("Instances")

    instances_dict = {
        "train": [],
        "test": [],
        "dev": [],
    }
    examples_dict = copy.deepcopy(instances_dict)

    if is_test:
        # zero-shot evaluation here, the test instances are the same across seeds!
        test_instances = all_instances[:100]
        all_instances = all_instances[100:]

        random.shuffle(all_instances)
        train_instances = all_instances[:num_examples]

        # some tasks won't have validation instances
        val_instances = all_instances[num_examples:2 * num_examples]
        if len(val_instances) < num_examples:
            print("Val instances insufficient", len(val_instances))
    else:
        val_instances = []
        test_instances = []
        random.shuffle(all_instances)

        # training instances
        train_instances = all_instances[:num_examples]
        val_instances = all_instances[num_examples:num_examples + 100]

        # minimum val instances
        if len(val_instances) <= 25:
            val_instances = all_instances[:25]
            train_instances = all_instances[25:25 + num_examples]

    instances_dict["train"] = train_instances
    instances_dict["dev"] = val_instances
    instances_dict["test"] = test_instances

    for key, instances in instances_dict.items():
        examples = []
        for instance in instances:
            example = task_data.copy()  # inject task data, cauz it's handier
            example["id"] = instance["id"]
            example["Instance"] = instance
            example["Task"] = task

            # pick one GT output for dev and test
            if key in ["dev", "test"]:
                output = random.choice(example["Instance"]["output"])
                example["Instance"]["output"] = [output]

            examples.append(example)
        examples_dict[key] = examples

    # print(task, len(examples_dict["train"]), len(examples_dict["dev"]))
    return {
        "task_name": task,
        "task_prefix": task,
        "train_examples": examples_dict["train"],
        "dev_examples": examples_dict["dev"],
        "test_examples": examples_dict["test"],
    }


@click.command
@click.argument('data_path', type=str)
@click.option('--seed', type=lambda x: x.split(","), default="13,42,58")
@click.option('--output_path', type=str)
@click.option('--num_examples', type=int, default=100)
def main(seed, num_examples, data_path, output_path):
    for s in seed:
        pl.seed_everything(s)
        out_path = os.path.join(f"{output_path}", f"{s}-{num_examples}")
        os.makedirs(out_path, exist_ok=True)

        with open(f"{data_path}/train_tasks.txt", "r") as r:
            train_tasks = [l.strip() for l in r.readlines()]

        i = 0
        for task in rich.progress.track(sorted(train_tasks), description="Train processing"):
            data = load_instances(task, data_path, num_examples)
            with open(f"{out_path}/{task}.json", "w") as f:
                json.dump(data, f)
            i += 1

        with open(f"{data_path}/test_tasks.txt", "r") as r:
            test_tasks = [l.strip() for l in r.readlines()]

        empty = 0.
        for task in rich.progress.track(sorted(test_tasks), description="Test processing"):
            # set test tasks task id to 0 as I fine-tune on each independently
            data = load_instances(task, data_path, num_examples, is_test=True)
            if not len(data["train_examples"]):
                empty += 1

            with open(f"{out_path}/{task}.json", "w") as f:
                json.dump(data, f)
        
        with open(f"{data_path}/excluded_tasks.txt", "r") as r:
            excluded_tasks = [l.strip() for l in r.readlines()]

        empty = 0.
        for task in rich.progress.track(sorted(excluded_tasks), description="Exc. processing"):
            # set test tasks task id to 0 as I fine-tune on each independently
            data = load_instances(task, data_path, num_examples, is_test=True)
            if not len(data["train_examples"]):
                empty += 1

            with open(f"{out_path}/{task}.json", "w") as f:
                json.dump(data, f)
        print("# empty tasks", empty)


if __name__ == '__main__':
    main()
