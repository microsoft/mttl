"""MMLU Dataset."""

import copy
import os

import datasets
import pandas as pd

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
mmlu_prompt_definition = (
    "The following are multiple choice questions (with answers) about {}.\n\n"
)


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


def _format_example_with_augmentation(
    prompt, options, label, include_answer=True, prefix="", suffix=""
):
    prompt = prefix + prompt + suffix
    for j in range(len(options)):
        prompt += "\n{}. {}".format(choices[j], options[j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(label)
    return prompt


def format_example_with_augmentation(
    example_prompt,
    example_options,
    example_label,
    icl_prompts: list,
    icl_options: list,
    icl_labels: list,
    prompt_def,
    augment_with_prompts,
    augment_with_options,
):
    """
    Using AI Harness prompt format discussed here: https://huggingface.co/blog/evaluating-mmlu-leaderboard
    This prepends "Choices:" keyword and "Question:" prefix

    Also creates varaitns with the right answer at varying positions in the option list
    """
    prompt_pos_augm = None
    prompt_pos = None

    if augment_with_prompts:
        prompt = _format_example_with_augmentation(
            example_prompt,
            example_options,
            example_label,
            prefix="Question:\n",
            suffix="\nChoices:",
            include_answer=False,
        )

        prompt_pos_augm = ""
        for icl_p, icl_o, icl_l in zip(icl_prompts, icl_options, icl_labels):
            prompt_pos_augm += _format_example_with_augmentation(
                icl_p, icl_o, icl_l, prefix="Question:\n", suffix="\nChoices:"
            )
        yield prompt, example_label, "", prompt_pos_augm

    if augment_with_options:
        label_idx = choices.index(example_label)
        for j, choice in enumerate(choices):
            if label_idx == j:
                continue
            _options = example_options.copy()
            # put the right answer in j's option
            _options[j], _options[label_idx] = _options[label_idx], _options[j]
            _label = choice
            prompt = _format_example_with_augmentation(
                example_prompt, _options, _label, include_answer=False
            )
            if prompt_pos is None:
                prompt_pos = ""
                for icl_p, icl_o, icl_l in zip(icl_prompts, icl_options, icl_labels):
                    prompt_pos += _format_example_with_augmentation(icl_p, icl_o, icl_l)
            yield prompt, _label, prompt_def, prompt_pos

            if augment_with_prompts:
                prompt = _format_example_with_augmentation(
                    example_prompt,
                    _options,
                    _label,
                    prefix="Question:\n",
                    suffix="\nChoices:",
                    include_answer=False,
                )
                if prompt_pos_augm is None:
                    prompt_pos_augm = ""
                    for icl_p, icl_o, icl_l in zip(
                        icl_prompts, icl_options, icl_labels
                    ):
                        prompt_pos_augm += _format_example_with_augmentation(
                            icl_p,
                            icl_o,
                            icl_l,
                            prefix="Question:\n",
                            suffix="\nChoices:",
                        )
                yield prompt, _label, "", prompt_pos_augm
    return


class MMLUConfig(datasets.BuilderConfig):
    def __init__(
        self,
        *args,
        name="default",
        description="Default config for MMLU",
        data_dir=None,
        task_dir=None,
        max_num_instances_per_task=None,
        max_num_instances_per_eval_task=None,
        augment_with_prompts=False,
        augment_with_option_permutations=False,
        **kwargs,
    ):
        super().__init__(
            name=name, description=description, data_dir=data_dir, *args, **kwargs
        )
        self.data_dir: str = data_dir
        self.task_dir: str = task_dir if task_dir else data_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.augment_with_prompts: bool = augment_with_prompts
        self.augment_with_option_permutations: bool = augment_with_option_permutations


class MMLUDataset(datasets.GeneratorBasedBuilder):
    """MMLU Dataset."""

    BUILDER_CONFIG_CLASS = MMLUConfig
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
                (
                    fn.split(f"_{subset}.csv")[0]
                    if f"_{subset}.csv" in fn
                    else fn.split(".csv")[0]
                )
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
                prompt_def = mmlu_prompt_definition.format(
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

                if subset != "auxiliary_train":
                    k = test_df.shape[1] - 2
                    options = [test_df.iloc[i, j + 1] for j in range(k)]
                    prompt = test_df.iloc[i, 0]

                    k = dev_df.shape[1] - 2
                    icl_prompts = [dev_df.iloc[j, 0] for j in range(ntrain)]
                    icl_options = [
                        [dev_df.iloc[j, s + 1] for s in range(k)] for j in range(ntrain)
                    ]
                    icl_labels = [dev_df.iloc[j, k + 1] for j in range(ntrain)]

                    for j, (_prompt_end, _label, _prompt_def, _prompt_pos) in enumerate(
                        format_example_with_augmentation(
                            prompt,
                            options,
                            label,
                            icl_prompts,
                            icl_options,
                            icl_labels,
                            prompt_def,
                            augment_with_prompts=self.config.augment_with_prompts,
                            augment_with_options=self.config.augment_with_option_permutations,
                        )
                    ):
                        instance = {
                            "Task": subject,
                            "Instance": {
                                "Input": _prompt_end,
                                "Output": _label,
                            },
                            "Definition": _prompt_def,
                            "Positive Examples": _prompt_pos,
                        }
                        yield f"{subject}_{i}_aug_{j}", instance
