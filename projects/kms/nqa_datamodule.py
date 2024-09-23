from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task


@dataclass
class NQADatasetConfig(DatasetConfig):
    task_name_field: str = "document_id"
    task_source_field: str = "document_id"
    prompt: str = "Answer the following question: "
    include_context: bool = False


@DataModule.register("narrativeqa", config_cls=NQADatasetConfig)
class NQADatamodule(DataModule):
    def setup_dataset(self):
        from mttl.models.library.dataset_library import DatasetLibrary

        dataset = DatasetLibrary.pull_dataset(self.config.dataset)

        (
            self._task_names,
            self._task_to_id,
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
        ) = maybe_filter_hf_dataset_by_task(
            dataset, self.config.task_name_field, self.config.finetune_task_name
        )

        def expand_questions(examples):
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
                    # take the first answer as the target
                    answer = examples["answers"][i][j][0]

                    if self.config.include_context:
                        source = [
                            {
                                "role": "user",
                                "content": f"Consider the following paragraph:\n{examples['text'][i]}\n{self.config.prompt}{question}",
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
                        self.tokenizer.apply_chat_template(
                            source, add_generation_prompt=True, tokenize=False
                        )
                    )
                    batch["target"].append(answer)
                    batch["answers"].append(examples["answers"][i][j])
                    batch["document_id"].append(examples["document_id"][i])
            return batch

        # create expanded versions of the dataset
        if self.train_dataset:
            self.train_dataset = self.train_dataset.map(
                expand_questions,
                batched=True,
                batch_size=1000,
                num_proc=16,
                remove_columns=self.train_dataset.column_names,
            )
        if self.dev_dataset:
            self.dev_dataset = self.dev_dataset.map(
                expand_questions,
                batched=True,
                batch_size=1000,
                num_proc=16,
                remove_columns=self.dev_dataset.column_names,
            )
        if self.test_dataset:
            self.test_dataset = self.test_dataset.map(
                expand_questions,
                batched=True,
                batch_size=1000,
                num_proc=16,
                remove_columns=self.test_dataset.column_names,
            )
