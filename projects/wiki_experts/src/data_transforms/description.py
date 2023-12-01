import json
from projects.wiki_experts.src.data_transforms.engines import AutoEngine
from src.data_transforms.base import (
    DataTransformTemplate,
    TransformConfig,
    TransformModel,
)
from dataclasses import dataclass
from src.data_transforms.utils import (
    upload_to_hf_,
)
import numpy as np
from datasets import load_dataset
import os


class DescriptionTemplate(DataTransformTemplate):
    @classmethod
    def apply(cls, context):
        template = """Here are few examples from a task. Please, analyze them carefully and then proceed to describe the task in natural language.

Here are the examples:
{}

Now, please, provide 3 diverse ways to describe the examples. Each description should capture what is common amongst the examples:
1.""".format(
            context
        )
        return template


@dataclass
class DescTransformConfig(TransformConfig):
    model_name: str = "gpt-35-turbo-instruct"
    num_examples: str = 5


class DescTransformModel(TransformModel):
    """Just a naive ID transform, where we just take the top N documents for each subject."""

    def __init__(
        self,
        config: DescTransformConfig,
    ):
        self.config = config

    def get_dataset_name(self):
        args = [
            f"desc",
            f"{self.config.model_name}",
        ]
        return "_".join(args) + ".json"

    def get_seed_dataset_(self, dataset_name, **options):
        """
        Convert a seed dataset of retrieved content into a tuple of (context, subject, icl_examples).
        """
        import tqdm

        dataset = load_dataset(dataset_name)["train"]
        task_names = list(set(list(dataset["task_name"])))

        task_examples = {task_name: [] for task_name in task_names}
        task_names_all = dataset["task_name"]
        for i, name in enumerate(task_names_all):
            task_examples[name].append(i)

        converted_dataset = []
        for task_name in tqdm.tqdm(task_names):
            examples = task_examples[task_name]

            for i in range(3):
                few_shot_examples = np.random.choice(
                    np.arange(len(examples)), self.config.num_examples
                )
                few_shot_context = "\n\n".join(
                    [
                        "\n".join(
                            [
                                "Input: "
                                + dataset[examples[few_shot_example]]["source"],
                                "Target: "
                                + dataset[examples[few_shot_example]]["target"],
                            ]
                        )
                        for few_shot_example in few_shot_examples
                    ]
                )

                if len(few_shot_context.split()) > 15000:
                    print("Cutting context down...")
                    few_shot_context = "\n\n".join(
                        [
                            "\n".join(
                                [
                                    "Input: "
                                    + dataset[examples[few_shot_example]]["source"],
                                    "Target: "
                                    + dataset[examples[few_shot_example]]["target"],
                                ]
                            )
                            for few_shot_example in few_shot_examples[:1]
                        ]
                    )

                converted_dataset.append(
                    {
                        "examples": DescriptionTemplate.apply(few_shot_context),
                        "task_name": task_name,
                    }
                )
        return converted_dataset

    def transform(
        self,
        dataset_name,
        upload_to_hub=False,
        output_path="./generated.jsonl",
        **kwargs,
    ):
        output_path = os.environ.get("AMLT_OUTPUT_DIR", output_path)
        if upload_to_hub:
            assert (
                os.environ.get("HF_TOKEN") is not None
            ), "Please set HF_TOKEN env variable."

        # start dataset
        prev_dataset = self.get_seed_dataset_(dataset_name)
        llm = AutoEngine.from_path(self.config.model_name)

        dataset = []
        with open(self.get_dataset_name(), "w") as f:
            for i, description in enumerate(
                llm.generate(
                    [ex["examples"] for ex in prev_dataset], stream=True, **kwargs
                )
            ):
                description = description[0].split("\n")
                description = [description[0]] + [
                    desc[2:].strip() for desc in description[1:]
                ]
                dataset.extend(
                    [
                        {
                            "task_name": prev_dataset[i]["task_name"],
                            "description": desc.strip(),
                        }
                        for desc in description
                        if desc
                    ]
                )
                print(description)
                f.write(json.dumps(dataset[i]) + "\n")

        if upload_to_hub:
            upload_to_hf_(self.get_dataset_name(), configuration=self.config)
