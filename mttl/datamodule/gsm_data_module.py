import os
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.models.library.dataset_library import DatasetLibrary
import json


@dataclass
class GsmDataConfig(DatasetConfig):
    pass


# code refer to https://github.com/aksh555/LoRA-Soups/blob/main/evaluate.py#L208
def generate_math_prompt(instruction, input=None):
    with open("mttl/datamodule/math.json", "r") as f:
        cot_data = json.load(f)
    prompt = """Let's use Python to solve math problems step by step. Below are a few Instruction-Response pairs on how to do it."""
    prompt += "\n\n"
    for data in cot_data:
        prompt += f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['output']}\n\n"
    prompt += "Now write a function 'solution' encolsed in ``` in Python to solve this Instruction. Write only a code block. Write only valid Python code without using any units with the numerical values and any invalid symbols.\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt


def instruct_templete(example):
    example["source"] = generate_math_prompt(example["input"])
    example["target"] = str(example["answer"])
    return example


@DataModule.register("gsm", config_cls=GsmDataConfig)
class GsmDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 4))
        dataset = DatasetLibrary.pull_dataset("reasoning-machines/gsm-hard")
        dataset = dataset.rename_column("target", "answer")
        dataset = dataset.map(instruct_templete, num_proc=n_proc)
        self.train_dataset = dataset["train"]
        self.dev_dataset = self.test_dataset = dataset["train"]


if __name__ == "__main__":
    config = GsmDataConfig(model="microsoft/Phi-3-mini-4k-instruct")

    datamodule = GsmDataModule(config, for_generation=True)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
