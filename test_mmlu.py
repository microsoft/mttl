from datasets import load_dataset
import pkg_resources

filename = pkg_resources.resource_filename(__name__, "mttl/dataloader/mmlu_dataset.py")
breakpoint()
datadir = "mmlu/data"

dataset = load_dataset(
    filename,
    data_dir=datadir,
)
breakpoint()
