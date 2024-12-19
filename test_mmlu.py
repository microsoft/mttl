from datasets import load_dataset
import pkg_resources

filename = pkg_resources.resource_filename(__name__, "mttl/dataloader/mmlu_dataset.py")

datadir = "mmlu/data"

dataset = load_dataset(
    filename,
    data_dir=datadir,
)

from mttl.datamodule.mmlu_data_module import MMLUDataModule, MMLUDataConfig


mmlu = MMLUDataModule(
    MMLUDataConfig(
        "mmlu",
        model="t5-small",
        model_family="seq2seq",
        train_batch_size=4,
        predict_batch_size=4,
    )
)

breakpoint()
