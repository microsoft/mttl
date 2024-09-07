import os
from collections import defaultdict
from functools import partial

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.logging import logger
from mttl.models.library.dataset_library import DatasetLibrary


class KMSummaryDatasetConfig(DatasetConfig):
    # prompt to summarize the text
    summarize_prompt: str = "Summarize the preceding passage."
    # load summaries from the dataset
    num_summaries: int = 4
    # field in the dataset that contains the task name
    task_name_field: str = "subject"
    # field in the dataset that contains the task source
    task_source_field: str = "subject"


class KMSummaryDataCollator(DefaultCollator):
    def __call__(self, batch):
        output_batch = super().__call__(batch)

        prompts = [b["prompt"] for b in batch]
        labels = [b["target"] for b in batch]
        prompt_batch = self.prepare_inputs_for_gpt_family(prompts, labels)

        # no context tensors used for context distillation loss
        output_batch["nc_input_ids"] = prompt_batch["input_ids"]
        output_batch["nc_attention_mask"] = prompt_batch["attention_mask"]
        output_batch["nc_labels"] = prompt_batch["labels"]
        return output_batch


@DataModule.register("dcd_summary", config_cls=KMSummaryDatasetConfig)
class KMSummaryDatasetModule(DataModule):
    @property
    def collate_class(self):
        return KMSummaryDataCollator

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        assert "train" in dataset

        def filter_summaries(example, n):
            return {"summaries": example["summaries"][:n]}

        logger.info(f"Keeping only {self.config.num_summaries} summaries per document.")
        dataset = dataset.map(
            partial(filter_summaries, n=self.config.num_summaries), num_proc=20
        )

        def expand_summaries_and_chat(example):
            return_dict = {
                "source": [],
                "target": [],
                "prompt": [],
                "subject": [],
            }

            prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": self.config.summarize_prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            for i in range(len(example["text"])):
                text = example["text"][i]
                summaries = example["summaries"][i]
                subject = example["subject"][i]

                source = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": text + "\n\n" + self.config.summarize_prompt,
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                for summary in summaries:
                    return_dict["source"].append(source)
                    return_dict["target"].append(summary)
                    return_dict["prompt"].append(prompt)
                    return_dict["subject"].append(subject)
            return return_dict

        dataset = dataset.map(
            expand_summaries_and_chat,
            batched=True,
            batch_size=1000,
            desc="Applying chat template to text column.",
            remove_columns=["summaries", "text"],
        )

        (
            self._task_names,
            self._task_to_id,
            train_dataset,
            _,
            _,
        ) = maybe_filter_hf_dataset_by_task(
            dataset,
            self.config.task_name_field,
            self.config.finetune_task_name,
            n_proc=n_proc,
        )

        self.train_dataset, self.dev_dataset = self.create_train_valid_split(
            train_dataset
        )
        self.test_dataset = self.dev_dataset
