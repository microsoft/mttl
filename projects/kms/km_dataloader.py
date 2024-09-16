import os
from collections import defaultdict
from functools import partial

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.datamodule.utils import maybe_filter_hf_dataset_by_task
from mttl.logging import logger
from mttl.models.library.dataset_library import DatasetLibrary

AVAILABLE_PROMPTS = {
    "summary": "Summarize the preceding passage.",
    "qa": "Generate a question-answer pair given the preceding passage.",
}


class KMDatasetConfig(DatasetConfig):
    # there might be multiple types, i.e. "qa", "summary", or maybe else in the future
    use_only_type: str = None
    # for each input example, we could have several outputs (e.g. several summaries or QA pairs), we can only use N of these
    num_outputs_per_chunk: int = 4
    # field in the dataset that contains the task name
    task_name_field: str = "subject"
    # field in the dataset that contains the task source
    task_source_field: str = "subject"


class KMDataCollator(DefaultCollator):
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


@DataModule.register("dcd_km", config_cls=KMDatasetConfig)
class KMDatasetModule(DataModule):
    @property
    def collate_class(self):
        return KMDataCollator

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry(self.config.dataset)
        n_proc = int(os.environ.get("MTTL_NUM_PROC_DATASETS", 16))

        assert "train" in dataset

        def filter_targets(example, n):
            return {"outputs": example["outputs"][:n]}

        logger.info(
            f"Keeping only {self.config.num_outputs_per_chunk} outputs per document."
        )
        dataset = dataset.map(
            partial(filter_targets, n=self.config.num_outputs_per_chunk), num_proc=20
        )

        if self.config.use_only_type:
            # filter types (e.g. use only summary, or use only qas)
            def filter_types(example, types):
                return example["type"] in types.split(",")

            dataset = dataset.filter(
                partial(filter_types, type=self.config.use_only_type), num_proc=20
            )

        def expand_targets_and_chat(example):
            return_dict = {
                "source": [],
                "target": [],
                "prompt": [],
                "subject": [],
            }

            for i in range(len(example["input"])):
                input = example["input"][i]
                outputs = example["outputs"][i]
                type = example["type"][i]
                subject = example["subject"][i]

                prompt = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": AVAILABLE_PROMPTS[type],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                source = self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": input + "\n\n" + AVAILABLE_PROMPTS[type],
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                for output in outputs:
                    return_dict["source"].append(source)
                    return_dict["target"].append(output)
                    return_dict["prompt"].append(prompt)
                    return_dict["subject"].append(subject)
            return return_dict

        dataset = dataset.map(
            expand_targets_and_chat,
            batched=True,
            batch_size=1000,
            desc="Applying chat template...",
            remove_columns=["input", "outputs", "type"],
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
