from datasets import load_dataset
from mttl.dataloader.alpaca_dataset_readers import AlpacaDataset


class PlatypusDataset(AlpacaDataset):
    def __init__(
        self,
    ):
        super().__init__()

        # load the data
        self.dataset = load_dataset("garage-bAInd/Open-Platypus")["train"]
