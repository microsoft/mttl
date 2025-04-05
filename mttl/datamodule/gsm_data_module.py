import os
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.models.library.dataset_library import DatasetLibrary
from functools import partial
import json


@dataclass
class GsmDataConfig(DatasetConfig):
    gsm_template: str = (
        "cot"  # the template we will use for the prompt, for code generation or chain of thought.
    )
    few_shot: int = 8  # number of fewshot examples to include in the prompt


# code refer to https://github.com/aksh555/LoRA-Soups/blob/main/evaluate.py#L208
def generate_math_prompt_with_python(instruction, input=None, few_shot=8):

    with open("mttl/datamodule/math.json", "r") as f:
        cot_data = json.load(f)
    prompt = """Let's use Python to solve math problems step by step. Below are a few Instruction-Response pairs on how to do it."""
    prompt += "\n\n"
    for data in cot_data:
        prompt += f"### Instruction:\n{data['instruction']}\n\n### Response:\n{data['output']}\n\n"
    prompt += "Now write a function 'solution' encolsed in ``` in Python to solve this Instruction. Write only a code block. Write only valid Python code without using any units with the numerical values and any invalid symbols.\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt


def instruct_template_python(example, few_shot):
    example["source"] = generate_math_prompt_with_python(
        example["input"], few_shot=few_shot
    )
    example["target"] = str(example["answer"])
    return example

def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Question: "
                + question[i]
                + "\nAnswer: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt

def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def instruct_template_cot(example):

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


@DataModule.register("gsm-8k", config_cls=GsmDataConfig)
class Gsm8kDataModule(DataModule):
    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset("openai/gsm8k")
        dataset = dataset['main']['test']
        dataset = dataset.rename_column("target", "answer")
        if self.config.gsm_template == "cot":
            dataset = dataset.map(instruct_template_cot, num_proc=n_proc)
        elif self.config.gsm_template == "python":
            partial_parse_function = partial(
                instruct_template_python, few_shot=self.config.few_shot
            )
            dataset = dataset.map(
                partial_parse_function,
                num_proc=n_proc,
            )
        self.train_dataset = dataset["train"]
        self.dev_dataset = self.test_dataset = dataset["train"]

@DataModule.register("gsm-8k-hard", config_cls=GsmDataConfig)
class Gsm8kHardDataModule(DataModule):
    def setup_dataset(self):
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 4))
        dataset = DatasetLibrary.pull_dataset("reasoning-machines/gsm-hard")
        dataset = dataset.rename_column("target", "answer")
        if self.config.gsm_template == "cot":
            dataset = dataset.map(instruct_template_cot, num_proc=n_proc)
        elif self.config.gsm_template == "python":
            partial_parse_function = partial(
                instruct_template_python, few_shot=self.config.few_shot
            )
            dataset = dataset.map(
                partial_parse_function,
                num_proc=n_proc,
            )
        self.train_dataset = dataset["train"]
        self.dev_dataset = self.test_dataset = dataset["train"]


if __name__ == "__main__":
    config = GsmDataConfig(
        model="microsoft/Phi-3-mini-4k-instruct", gsm_template="python", few_shot=1
    )

    datamodule = Gsm8kDataModule(config, for_generation=True)
    train_dataloader = Gsm8kDataModule.train_dataloader()
    val_dataloder = datamodule.val_dataloader()
    for batch in val_dataloder:
        print(batch)
        breakpoint()
