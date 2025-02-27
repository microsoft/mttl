import json
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, MultiChoiceDataModule
from mttl.datamodule.utils import (
    apply_custom_split_file,
    maybe_filter_hf_dataset_by_task,
    split_on_split_column,
)


@dataclass
class LonghealthDatasetConfig(DatasetConfig):
    task_name_field: str = "document_id"
    task_source_field: str = "document_id"
    prompt: str = (
        """
You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. When you receive one or more of these letters, you will be expected to carefully review the contents and accurately answer multiple-choice questions related to these documents. 

Your answers should be:
1. Accurate: Make sure your answers are based on the information provided in the letters.
2. Concise: Provide brief and direct answers without unnecessary elaboration.
3. Contextual: Consider the context and specifics of each question to provide the most relevant information.

Remember, your job is to streamline the physician's decision-making process by providing them with accurate and relevant information from discharge summaries. Efficiency and reliability are key.
"""
    )
    include_context: bool = False
    topk_context: int = 3
    include_all_answers: bool = True


@DataModule.register("longhealth", config_cls=LonghealthDatasetConfig)
class LonghealthDatamodule(MultiChoiceDataModule):
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
                                    f"Document {k+1}: {context[k]}\n\n"
                                    for k in range(
                                        min(self.config.topk_context, len(context))
                                    )[::-1]
                                ]
                            )
                            context = (
                                "--------------BEGIN DOCUMENTS--------------\n\n"
                                + context
                                + "--------------END DOCUMENTS--------------\n\n"
                            )
                            source = [
                                {
                                    "role": "user",
                                    "content": f"{self.config.prompt}{context} Question: {question}",
                                }
                            ]
                        else:
                            context = (
                                "--------------BEGIN DOCUMENTS--------------\n\n"
                                + context
                                + "--------------END DOCUMENTS--------------\n\n"
                            )
                            source = [
                                {
                                    "role": "user",
                                    "content": f"{self.config.prompt}{context} Question: {question}",
                                }
                            ]
                    else:
                        source = [
                            {
                                "role": "user",
                                "content": f"{self.config.prompt} Question: {question}",
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
