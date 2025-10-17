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
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
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

        self.train_dataset = self.train_dataset.map(
            lambda examples: expand_questions(examples, self.tokenizer),
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=self.train_dataset.column_names,
        )
        if self.dev_dataset:
            self.dev_dataset = self.dev_dataset.map(
                lambda examples: expand_questions(examples, self.tokenizer),
                batched=True,
                batch_size=1000,
                num_proc=1,
                remove_columns=self.dev_dataset.column_names,
            )
        else:
            self.dev_dataset = self.train_dataset

        self.test_dataset = self.dev_dataset