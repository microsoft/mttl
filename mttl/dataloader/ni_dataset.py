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

import datasets

from mttl.utils import logger

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions.
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting.
These tasks are collected with contributions of NLP practitioners in the community and
through an iterative peer review process to ensure their quality.
"""


_URL = "https://instructions.apps.allenai.org/"


class NIConfig(datasets.BuilderConfig):
    def __init__(
        self,
        *args,
        name="default",
        description="Default config for NaturalInstructions",
        data_dir=None,
        task_dir=None,
        task_name=None,
        max_num_instances_per_task=None,
        max_num_instances_per_eval_task=None,
        **kwargs,
    ):
        super().__init__(
            name=name, description=description, data_dir=data_dir, *args, **kwargs
        )
        self.task_name: str = task_name
        self.data_dir: str = data_dir
        self.task_dir: str = task_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task


class NIOriginalDataset(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [
                        {
                            "input": datasets.Value("string"),
                            "output": datasets.Value("string"),
                            "explanation": datasets.Value("string"),
                        }
                    ],
                    "Negative Examples": [
                        {
                            "input": datasets.Value("string"),
                            "output": datasets.Value("string"),
                            "explanation": datasets.Value("string"),
                        }
                    ],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")],
                    },
                    "Instance License": [datasets.Value("string")],
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("data_dir cannot be None.")

        data_dir = os.path.join(self.config.data_dir, "splits", "default")
        task_dir = os.path.join(self.config.data_dir, "tasks")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(data_dir, "train_tasks.txt"),
                    "task_dir": task_dir,
                    "task_name": self.config.task_name,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(data_dir, "test_tasks.txt"),
                    "task_dir": task_dir,
                    "task_name": self.config.task_name,
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(data_dir, "test_tasks.txt"),
                    "task_dir": task_dir,
                    "task_name": self.config.task_name,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test",
                },
            ),
        ]

    def _generate_examples(
        self,
        path=None,
        task_dir=None,
        task_name=None,
        max_num_instances_per_task=None,
        subset=None,
    ):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")

        with open(path, encoding="utf-8") as split_f:
            for line in split_f:
                task_name_ = line.strip()

                if task_name is not None and task_name_ != task_name:
                    continue
                else:
                    task_name = task_name_

                task_path = os.path.join(task_dir, task_name + ".json")

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
                    elif subset == "validation":
                        instances = all_instances[100:]
                    elif subset == "train":
                        instances = all_instances
                    if (
                        subset != "test"
                        and max_num_instances_per_task is not None
                        and max_num_instances_per_task >= 0
                    ):
                        random.shuffle(instances)
                        instances = instances[:max_num_instances_per_task]
                    for idx, instance in enumerate(instances):
                        example = task_data.copy()
                        example["id"] = instance["id"]
                        example["Instance"] = instance
                        yield f"{task_name}_{idx}", example
