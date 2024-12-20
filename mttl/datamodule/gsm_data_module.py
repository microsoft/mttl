import os
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.models.library.dataset_library import DatasetLibrary
import json


@dataclass
class GsmDataConfig(DatasetConfig):
    templete: str = (
        "cot"  # the templete we will use for the prompt, for code generation or chain of thought.
    )


# code refer to https://github.com/aksh555/LoRA-Soups/blob/main/evaluate.py#L208
def generate_math_prompt_with_python(instruction, input=None):
    with open("mttl/datamodule/math.json", "r") as f:
        cot_data = json.load(f)
    prompt = """Let's use Python to solve math problems step by step. Below are a few Instruction-Response pairs on how to do it."""
    prompt += "\n\n"
    for data in cot_data:
        prompt += f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['output']}\n\n"
    prompt += "Now write a function 'solution' encolsed in ``` in Python to solve this Instruction. Write only a code block. Write only valid Python code without using any units with the numerical values and any invalid symbols.\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt


def instruct_templete_python(example):
    example["source"] = generate_math_prompt_with_python(example["input"])
    example["target"] = str(example["answer"])
    return example


def instruct_templete_cot(example):

    PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""
    PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

    Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
    A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

    Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
    A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

    Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
    A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

    Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
    A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

    Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
    A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

    Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
    A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

    Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
    A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""

    TEMPLATE = """
    Q: {question}
    A:"""

    full_prompt = (
        PREAMBLE + "\n\n" + PROMPT + "\n" + TEMPLATE.format(question=example["input"])
    )
    example["source"] = full_prompt
    example["target"] = str(example["answer"])
    return example


@DataModule.register("gsm", config_cls=GsmDataConfig)
class GsmDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 4))
        dataset = DatasetLibrary.pull_dataset("reasoning-machines/gsm-hard")
        dataset = dataset.rename_column("target", "answer")
        if self.config.templete == "cot":
            dataset = dataset.map(instruct_templete_cot, num_proc=n_proc)
        elif self.config.templete == "python":
            dataset = dataset.map(instruct_templete_python, num_proc=n_proc)
        self.train_dataset = dataset["train"]
        self.dev_dataset = self.test_dataset = dataset["train"]


if __name__ == "__main__":
    config = GsmDataConfig(model="microsoft/Phi-3-mini-4k-instruct", templete="cot")

    datamodule = GsmDataModule(config, for_generation=True)
    train_dataloader = datamodule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
