from dataclasses import dataclass


from mttl.datamodule.base import DataModule, DatasetConfig
from mttl.models.library.dataset_library import DatasetLibrary


@dataclass
class AdvBenchDataModuleConfig(DatasetConfig):
    pass


class AdvBenchDataModule(DataModule):

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry("walledai/AdvBench")
        dataset = dataset["train"]
        dataset = dataset.rename_column("prompt", "source")

        def map_example(example):
            user_prompt = example["source"]
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {"source": prompt, "target": example["target"]}

        if self.tokenizer.chat_template is not None:
            dataset = dataset.map(map_example)
        self.train_dataset = dataset
        self.dev_dataset = self.test_dataset = dataset


if __name__ == "__main__":
    config = AdvBenchDataModuleConfig(model="microsoft/Phi-3-mini-4k-instruct")
    data_module = AdvBenchDataModule(config, for_generation=True)
    train_dataloader = data_module.train_dataloader()
    for batch in train_dataloader:
        print(batch)
        break
