from dataclasses import dataclass

from mttl.datamodule.base import DataModule, DatasetConfig, DefaultCollator
from mttl.models.library.dataset_library import DatasetLibrary
from mttl.datamodule.prompts import get_task_instruction_math
@dataclass
class Math500DataConfig(DatasetConfig):
    pass


@dataclass
class Math500DataModule(DataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset_with_retry("HuggingFaceH4/MATH-500")['test']

        def map_example(example):
            user_prompt = get_task_instruction_math(example["source"], model_name='qwq')
            prompt = [{"role": "user", "content": user_prompt}]
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            return {"source": prompt, "target": example["target"]}

        dataset = dataset.rename_column("problem", "source")
        dataset = dataset.rename_column("answer", "target")
        if self.tokenizer.chat_template is not None:
            dataset = dataset.map(map_example, num_proc=1)
        self.train_dataset = self.test_dataset = self.dev_dataset = dataset

        self.print_infos()

if __name__ == "__main__":
    config = Math500DataConfig(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    datamodule = Math500DataModule(config)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    for batch in val_dataloader:
        print(batch)
        breakpoint()
