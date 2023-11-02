import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mttl.dataloader.mmlu_dataset import MMLUDataset


def test_MMLUDataset():
    dataset = MMLUDataset()
    dataset.config.augment_with_prompts = True
    dataset.config.augment_with_option_permutations = True
    gen = dataset._generate_examples(
        data_dir=os.environ["MMLU_DATA_DIR"],
        subset="val",
        max_num_instances_per_task=10,
    )
    for i in range(10):
        next(gen)

    dataset.config.augment_with_prompts = False
    dataset.config.augment_with_option_permutations = False
    gen = dataset._generate_examples(
        data_dir=os.environ["MMLU_DATA_DIR"],
        subset="val",
        max_num_instances_per_task=10,
    )
    for i in range(10):
        next(gen)

    dataset.config.augment_with_prompts = True
    dataset.config.augment_with_option_permutations = False
    gen = dataset._generate_examples(
        data_dir=os.environ["MMLU_DATA_DIR"],
        subset="val",
        max_num_instances_per_task=10,
    )
    for i in range(10):
        next(gen)

    dataset.config.augment_with_prompts = False
    dataset.config.augment_with_option_permutations = True
    gen = dataset._generate_examples(
        data_dir=os.environ["MMLU_DATA_DIR"],
        subset="val",
        max_num_instances_per_task=10,
    )
    for i in range(10):
        next(gen)
