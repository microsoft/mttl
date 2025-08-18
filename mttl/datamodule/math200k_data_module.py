from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.models.library.dataset_library import DatasetLibrary
from mttl.datamodule.prompts import get_task_instruction_math

@dataclass
class Math200kDataConfig(DatasetConfig):
    pass


@dataclass
class Math200kDataModule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):    
        dataset = DatasetLibrary.pull_dataset_with_retry("open-r1/OpenR1-Math-220k")
        dataset = dataset['train']
        def map_example(example):
            # Use tokenizer.apply_chat_template() to format the conversation
            messages = [
                {
                    "role": "user",
                    "content": example["source"],
                },
                {
                    "role": "assistant",
                    "content": example["target"],
                },
            ]
            input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            if "deepseek" in self.config.model:
                source = input.split("<｜Assistant｜>")[0] + "<｜Assistant｜>"
                target = input.split("<｜Assistant｜>")[1]
            else:
                source = input.split("<|assistant|>")[0] + "<|assistant|>"
                target = input.split("<|assistant|>")[1]
            return {"source": source, "target": target}
        # they have their own source
        dataset = dataset.rename_column("source","dataset_source")
        dataset = dataset.rename_column("problem", "source") 
        dataset = dataset.rename_column("solution", "target")
        if self.tokenizer.chat_template is not None:
            dataset = dataset.map(map_example, num_proc=1)
        self.train_dataset = self.test_dataset = self.dev_dataset = dataset

        self.print_infos()

    @property
    def collate_fn(self):
        return DefaultCollator(
            tokenizer=self.tokenizer,
            padding="longest", 
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            return_tensors="pt",
            model_family=self.config.model_family,
            for_generation=self.for_generation,
        )


if __name__ == "__main__":
    config = Math200kDataConfig(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    datamodule = Math200kDataModule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    for batch in val_dataloader:
        print(batch)
        breakpoint()
