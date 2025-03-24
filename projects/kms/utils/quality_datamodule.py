import json
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, MultiChoiceDataModule
from mttl.datamodule.utils import (
    apply_custom_split_file,
    maybe_filter_hf_dataset_by_task,
    split_on_split_column,
)


@dataclass
class QualityDatasetConfig(DatasetConfig):
    task_name_field: str = "document_id"
    task_source_field: str = "document_id"
    prompt: str = (
        "Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness. Question: "
    )
    include_context: bool = False
    topk_context: int = 10
    include_all_answers: bool = True


@DataModule.register("quality", config_cls=QualityDatasetConfig)
class QualityDatamodule(MultiChoiceDataModule):
    def setup_dataset(self):
        from mttl.models.library.dataset_library import DatasetLibrary

        dataset = DatasetLibrary.pull_dataset(self.config.dataset)

        # Instead of always working with the large datasets, we can subsample it
        if self.config.custom_split_file:
            dataset = apply_custom_split_file(dataset, self.config.custom_split_file)

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, self.config.task_name_field, self.config.finetune_task_name
        )

        def expand_questions(examples, tokenizer):
            batch = {
                "source": [],
                "target": [],
                "label_index": [],
                "document_id": [],
            }

            for i in range(len(examples["document_id"])):
                for j in range(len(examples["questions"][i])):
                    document_id = examples["document_id"][i]
                    question = examples["questions"][i][j]
                    options = examples["options"][i][j]
                    gold_label = examples["gold_label"][i][j]
                    if gold_label == -1:
                        gold_label = label_index = None
                    else:
                        label_index = gold_label - 1

                    if self.config.include_context:
                        context = examples["text"][i]
                        if isinstance(context, list):
                            # If the context is a list of strings per question, we get the question-specific context
                            context = context[j]

                        if isinstance(context, list):
                            # following Alan's approach
                            context = " ".join(
                                [
                                    f"Passage {k+1}: {context[k]}\n\n"
                                    for k in range(
                                        min(self.config.topk_context, len(context))
                                    )[::-1]
                                ]
                            )
                            source = [
                                {
                                    "role": "user",
                                    "content": f"Consider the following passages:\n{context}\n{self.config.prompt}{question}",
                                }
                            ]
                        else:
                            source = [
                                {
                                    "role": "user",
                                    "content": f"Consider the following paragraph:\n{context}\n{self.config.prompt}{question}",
                                }
                            ]
                    else:
                        source = [
                            {
                                "role": "user",
                                "content": f"{self.config.prompt}{question}",
                            }
                        ]

                    if self.config.include_all_answers:
                        batch["source"].append(
                            tokenizer.apply_chat_template(
                                source, add_generation_prompt=True, tokenize=False
                            )
                        )
                        batch["target"].append(options)
                        batch["label_index"].append(label_index)
                    else:
                        batch["source"].append(
                            tokenizer.apply_chat_template(
                                source, add_generation_prompt=True, tokenize=False
                            )
                        )
                        if label_index is None:
                            batch["target"].append(None)
                            batch["label_index"].append(-1)
                        else:
                            batch["target"].append([options[label_index]])
                            batch["label_index"].append(0)
                    batch["document_id"].append(examples["document_id"][i])
            return batch

        if self.tokenizer.chat_template is None:
            self.tokenizer.apply_chat_template = lambda x, **kwargs: x[0]["content"]

        if "split" in train_dataset.features:
            self.train_dataset, self.dev_dataset, self.test_dataset = (
                split_on_split_column(train_dataset)
            )
            self.train_dataset = self.train_dataset.map(
                lambda examples: expand_questions(examples, self.tokenizer),
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=train_dataset.column_names,
            )
            self.dev_dataset = self.dev_dataset.map(
                lambda examples: expand_questions(examples, self.tokenizer),
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=train_dataset.column_names,
            )
            self.test_dataset = self.dev_dataset
        else:
            train_dataset = train_dataset.map(
                lambda examples: expand_questions(examples, self.tokenizer),
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=train_dataset.column_names,
            )
            self.train_dataset = self.dev_dataset = self.test_dataset = train_dataset


prompt_template_w_docs = """
--------------BEGIN CONTEXT--------------

{documents}

--------------END CONTEXT--------------

{question_text}
{options}

Please answer using the following format:
0. Begin your answer with the phrase "The correct answer is".
1. State the letter of the correct option (e.g., A, B, C, D).
2. Follow the letter with a colon and the exact text of the option you chose.
3. Make sure your answer is a single, concise sentence.

For example, if the correct answer to a question is option C, and the text for C is 'Acute Bronchitis', your answer should be: 
'The correct answer is C: Acute bronchitis.'
"""

prompt_template_no_docs = """
{question_text}
{options}

Please answer using the following format:
1. Begin your answer with the phrase "The correct answer is".
2. State the letter of the correct option (e.g., A, B, C, D).
3. Follow the letter with a colon and the exact text of the option you chose.
4. Make sure your answer is a single, concise sentence.

For example, if the correct answer to a question is option C, and the text for C is 'Acute Bronchitis', your answer should be: 
'The correct answer is C: Acute bronchitis.'
"""

max_new_tokens = 50


@dataclass
class GenQualityDatasetConfig(DatasetConfig):
    task_name_field: str = "document_id"
    task_source_field: str = "document_id"
    prompt: str = (
        "Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness. Question: "
    )
    include_context: bool = False
    topk_context: int = 10
    include_all_answers: bool = True


@DataModule.register("gen_quality", config_cls=GenQualityDatasetConfig)
class GenQualityDataModule(DataModule):
    def setup_dataset(self):
        from mttl.models.library.dataset_library import DatasetLibrary

        dataset = DatasetLibrary.pull_dataset(self.config.dataset)

        # Instead of always working with the large datasets, we can subsample it
        if self.config.custom_split_file:
            dataset = apply_custom_split_file(dataset, self.config.custom_split_file)

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, self.config.task_name_field, self.config.finetune_task_name
        )

        # Let's make sure that the full prompt is always in context
        len_template = len(self.tokenizer.encode(prompt_template_w_docs))

        def expand_questions(examples, tokenizer, len_template):
            batch = {
                "source": [],
                "target": [],
                "document_id": [],
            }

            for i in range(len(examples["document_id"])):
                for j in range(len(examples["questions"][i])):
                    document_id = examples["document_id"][i]
                    question = examples["questions"][i][j]
                    options = examples["options"][i][j]
                    gold_label = examples["gold_label"][i][j]
                    if gold_label == -1:
                        gold_label = label_index = None
                    else:
                        label_index = gold_label - 1

                    """ NEW """
                    letters = ["A", "B", "C", "D"]
                    option_str = "\n".join(
                        [f"{letters[i]}: {option}" for i, option in enumerate(options)]
                    )
                    len_question = len(tokenizer.encode(question))
                    len_options = len(tokenizer.encode(option_str))
                    len_suffix = len(tokenizer.encode("The correct answer is: "))

                    total_len = len_question + len_options + len_template + len_suffix

                    if self.config.include_context:
                        context = examples["text"][i]
                        assert (
                            type(context) == str
                        ), f"Context should be a string, but got {type(context)}"

                        # Let's do some rough trucation if needed
                        context_ids = tokenizer.encode(context)
                        len_context = len(context_ids)
                        space_left = self.config.max_input_length - total_len

                        if space_left < len_context:
                            context_ids = context_ids[: max(0, space_left - 20)]
                            context = tokenizer.decode(
                                context_ids, skip_special_tokens=True
                            )

                        prompt = prompt_template_w_docs.format(
                            documents=context,
                            question_text=question,
                            options=option_str,
                        )
                    else:
                        prompt = prompt_template_no_docs.format(
                            question_text=question,
                            options=option_str,
                        )

                    """
                    source = [
                        {
                            "role": "system",
                            "content": sys_prompt,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ]
                    """
                    source = [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ]

                    batch["source"].append(
                        tokenizer.apply_chat_template(
                            source, add_generation_prompt=True, tokenize=False
                        )
                        + "The correct answer is"
                    )
                    batch["target"].append(
                        letters[label_index]
                    )  # [options[label_index]])
                    batch["document_id"].append(examples["document_id"][i])

            return batch

        if self.tokenizer.chat_template is None:
            self.tokenizer.apply_chat_template = lambda x, **kwargs: x[0]["content"]

        if "split" in train_dataset.features:
            self.train_dataset, self.dev_dataset, self.test_dataset = (
                split_on_split_column(train_dataset)
            )
            self.train_dataset = self.train_dataset.map(
                lambda examples: expand_questions(
                    examples, self.tokenizer, len_template
                ),
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=train_dataset.column_names,
            )
            self.dev_dataset = self.dev_dataset.map(
                lambda examples: expand_questions(
                    examples, self.tokenizer, len_template
                ),
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=train_dataset.column_names,
            )
            self.test_dataset = self.dev_dataset
        else:
            train_dataset = train_dataset.map(
                lambda examples: expand_questions(
                    examples, self.tokenizer, len_template
                ),
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=train_dataset.column_names,
            )
            self.train_dataset = self.dev_dataset = self.test_dataset = train_dataset
