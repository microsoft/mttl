"""MMLU Dataset."""

import pandas as pd
import os
import datasets

from mttl.utils import logger


_CITATION = """\
@article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }
"""

_DESCRIPTION = """\
This is a massive multitask test consisting of multiple-choice questions from various branches of knowledge, covering 57 tasks including elementary mathematics, US history, computer science, law, and more.
"""

_HOMEPAGE = "https://github.com/hendrycks/test"

_URL = "data.tar"


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def format_example_with_ai_harness_prompt(
    df, idx, label, include_answer, augment_with_prompts, augment_with_options
):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


# def format_example_with_ai_harness_prompt(df, idx, include_answer=True):
#     '''
#     Using AI Harness prompt format discussed here: https://huggingface.co/blog/evaluating-mmlu-leaderboard
#     This prepends "Choices:" keyword and "Question:" prefix
#     '''
#     prompt = "Question:\n" + df.iloc[idx, 0]
#     prompt+= "\nChoices:"
#     k = df.shape[1] - 2
#     for j in range(k):
#         prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
#     prompt += "\nAnswer:"
#     if include_answer:
#         prompt += " {}\n\n".format(df.iloc[idx, k + 1])
#     return prompt


class MMLUConfig(datasets.BuilderConfig):
    def __init__(
        self,
        *args,
        data_dir=None,
        task_dir=None,
        max_num_instances_per_task=None,
        max_num_instances_per_eval_task=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir: str = data_dir
        self.task_dir: str = task_dir if task_dir else data_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.augment_with_prompts: bool = True
        self.augment_with_option_permutations: bool = True


class MMLUDataset(datasets.GeneratorBasedBuilder):
    """MMLU Dataset."""

    BUILDER_CONFIG_CLASS = MMLUConfig
    BUILDER_CONFIGS = [
        MMLUConfig(name="default", description="Default config for MMLU")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Task": datasets.Value("string"),
                    "Instance": {
                        "Input": datasets.Value("string"),
                        "Output": datasets.Value("string"),
                    },
                    "Positive Examples": datasets.Value("string"),
                    "Definition": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": self.config.task_dir,
                    "subset": "auxiliary_train",
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_dir": self.config.task_dir,
                    "subset": "val",
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": self.config.task_dir,
                    "subset": "test",
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                },
            ),
        ]

    def _generate_examples(
        self, data_dir=None, subset=None, max_num_instances_per_task=None, ntrain=5
    ):
        subjects = sorted(
            [
                fn.split(f"_{subset}.csv")[0]
                if f"_{subset}.csv" in fn
                else fn.split(".csv")[0]
                for fn in os.listdir(os.path.join(data_dir, subset))
            ]
        )

        logger.info(f"Found subjects: {subjects}")

        for subject in subjects:
            if subset != "auxiliary_train":
                dev_df = pd.read_csv(
                    os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
                )[:ntrain]
            test_df = pd.read_csv(
                os.path.join(
                    data_dir,
                    subset,
                    subject
                    + (f"_{subset}" if subset != "auxiliary_train" else "")
                    + ".csv",
                ),
                header=None,
            )

            for i in range(test_df.shape[0]):
                if (
                    max_num_instances_per_task is not None
                    and max_num_instances_per_task >= 0
                ):
                    if i > max_num_instances_per_task:
                        break

                prompt_end = format_example(test_df, i, include_answer=False)
                prompt_def = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                    format_subject(subject).strip()
                )
                prompt_pos = ""
                if subset != "auxiliary_train":
                    for k in range(ntrain):
                        prompt_pos += format_example(dev_df, k)
                else:
                    prompt_pos = ""

                label = test_df.iloc[i, test_df.shape[1] - 1]

                instance = {
                    "Task": subject,
                    "Instance": {
                        "Input": prompt_end,
                        "Output": label,
                    },
                    "Definition": prompt_def,
                    "Positive Examples": prompt_pos,
                }
                yield f"{subject}_{i}", instance

                for j, (prompt_end, label, prompt_def) in enumerate(
                    format_example_with_ai_harness_prompt(
                        test_df, i, label, include_answer=True
                    )
                ):
                    instance = {
                        "Task": subject,
                        "Instance": {
                            "Input": prompt_end,
                            "Output": label,
                        },
                        "Definition": prompt_def,
                        "Positive Examples": prompt_pos,
                    }
                    yield f"{subject}_{i}_aug_{j}", instance

                # if self.config.augment_with_prompts:
                #     prompt_end_aug = format_example_with_ai_harness_prompt(test_df, i, include_answer=False)
                #     prompt_def_aug = "" # remove definition like in AI HArness case discussed here https://huggingface.co/blog/evaluating-mmlu-leaderboard
                #     instance = {'Task': subject,
                #                 'Instance': {
                #                     'Input': prompt_end_aug,
                #                     'Output': label,
                #                 },
                #                 'Definition': prompt_def_aug,
                #                 'Positive Examples': prompt_pos,
                #                 }
                #     yield f"{subject}_{i}_aug", instance

                # if self.config.augment_with_option_permutations:
