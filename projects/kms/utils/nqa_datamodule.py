import json
from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import (
    apply_custom_split_file,
    maybe_filter_hf_dataset_by_task,
    split_on_split_column,
)


@dataclass
class NQADatasetConfig(DatasetConfig):
    task_name_field: str = "document_id"
    task_source_field: str = "document_id"
    prompt: str = (
        "Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness. Question: "
    )
    expand_questions: bool = True
    include_context: bool = False
    topk_context: int = 10


@DataModule.register("narrativeqa", config_cls=NQADatasetConfig)
class NQADatamodule(DataModule):
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

        self.train_dataset, self.dev_dataset, self.test_dataset = split_on_split_column(
            train_dataset
        )

        def expand_questions(examples, tokenizer):
            def maybe_truncate(fixed_prompt, content, buffer=10):
                return content

            batch = {
                "source": [],
                "target": [],
                "answers": [],
                "document_id": [],
            }
            for i in range(len(examples["document_id"])):
                for j in range(len(examples["questions"][i])):
                    document_id = examples["document_id"][i]
                    question = examples["questions"][i][j]

                    if self.for_generation:
                        answer = examples["answers"][i][j]
                    else:
                        # take the first answer as the target
                        answer = examples["answers"][i][j][0]

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
                            context = maybe_truncate(
                                f"Consider the following passages:\n\n{self.config.prompt}{question}",
                                context,
                            )
                            source = [
                                {
                                    "role": "user",
                                    "content": f"Consider the following passages:\n{context}\n{self.config.prompt}{question}",
                                }
                            ]
                        else:
                            context = maybe_truncate(
                                f"Consider the following paragraph:\n\n{self.config.prompt}{question}",
                                context,
                            )
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

                    batch["source"].append(
                        tokenizer.apply_chat_template(
                            source, add_generation_prompt=True, tokenize=False
                        )
                        if tokenizer.chat_template is not None
                        else source[0]["content"]
                    )
                    batch["target"].append(answer)
                    batch["answers"].append(examples["answers"][i][j])
                    batch["document_id"].append(examples["document_id"][i])
            return batch

        for split in ["train", "dev", "test"]:
            dataset = getattr(self, f"{split}_dataset")

            if dataset and self.config.expand_questions:
                dataset = dataset.map(
                    lambda examples: expand_questions(examples, self.tokenizer),
                    batched=True,
                    batch_size=1000,
                    num_proc=20,
                    remove_columns=dataset.column_names,
                )
                setattr(self, f"{split}_dataset", dataset)
