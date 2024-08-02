from mttl.datamodule.base import DatasetConfig, DefaultDataModule
from mttl.models.library.expert_library import DatasetLibrary


class ChatDataConfig(DatasetConfig):
    chat_template: str = None  # TODO: load and apply custom chat template
    seed: str = 42


class ChatDataModule(DefaultDataModule):

    def setup_dataset(self):
        dataset = DatasetLibrary.pull_dataset(self.config.dataset, split="train")

        num_examples = len(dataset)
        num_train = int(0.8 * num_examples)
        num_dev = int(0.1 * num_examples)

        dataset = dataset.shuffle(seed=self.config.seed)

        dataset = dataset.rename_column("task_name", "origin_task_name")
        dataset = dataset.rename_column("cluster_id", "task_name")
        self._task_names = sorted(list(set(dataset["task_name"])))
        self._task_to_id = {
            task_name: i for i, task_name in enumerate(self._task_names)
        }

        self.train_dataset = dataset.select(range(num_train))
        self.dev_dataset = dataset.select(range(num_train, num_train + num_dev))
        self.test_dataset = dataset.select(range(num_train + num_dev, num_examples))
