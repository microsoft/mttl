# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
import random
import tqdm
import datasets
import torch

logger = datasets.logging.get_logger(__name__)


class NIOriginalDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class NIOriginalDatasetReader():
    """NaturalInstructions Dataset."""
    def __init__(
        self,
        data_dir=None,
        max_num_instances_per_task=100,
        **kwargs,
    ):
        self.data_dir: str = data_dir
        self.task_dir: str = f"{data_dir}/tasks"
        self.task_names: dict = {"train": set(), "valid": set(), "test": set()}
        self.max_num_instances_per_task: int = max_num_instances_per_task

    def read_orig_datasets(self, subset=None):
        """Yields examples."""
        examples = []

        if subset in ["valid", "test"]:
            path = "test_tasks.txt"
        elif subset in ["train"]:
            path = "train_tasks.txt"

        data_dir = f"{self.data_dir}/{path}"

        with open(data_dir, encoding="utf-8") as split_f:
            for line in tqdm.tqdm(split_f, desc=f"Reading tasks from {subset}"):
                task_name = line.strip()

                self.task_names[subset].add(task_name)
                task_path = os.path.join(self.task_dir, task_name + ".json")

                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    task_data["Task"] = task_name
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    all_instances = task_data.pop("Instances")
                    if subset == "test":
                        # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                        # we put them in the first for reproducibility.
                        # so, we use them here
                        instances = all_instances[:100]
                    elif subset == "valid":
                        instances = all_instances[100:]
                    elif subset == "train":
                        instances = all_instances
                    if (
                        self.max_num_instances_per_task is not None
                        and self.max_num_instances_per_task >= 0
                    ):
                        random.shuffle(instances)
                        instances = instances[:self.max_num_instances_per_task]

                    for idx, instance in enumerate(instances):
                        example = task_data.copy()
                        example["id"] = instance["id"]
                        example["Instance"] = instance
                        examples.append(example)
        return NIOriginalDataset(examples)
