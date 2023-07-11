import os
import click

from mttl.datamodule.t0_data_module import T0PretrainDataModule
from mhr_config import MHRConfig as Config


# T-Few github repo, to get original T0 splits
T_FEW_GITHUB_REPO = "https://github.com/r-three/t-few/"


@click.command()
@click.option('--output_path', type=str)
def main(output_path):
    os.makedirs(output_path, exist_ok=True)

    # clone t-few repo
    print("Cloning ", T_FEW_GITHUB_REPO)
    os.system("git clone " + T_FEW_GITHUB_REPO + " /tmp/t-few/")

    # move few-shot test sets to data_path
    print("Moving few shot sets...")
    os.system("cp -rf /tmp/t-few/data/few_shot/ " + output_path + "/few_shot/")

    t0_config = Config(
        filenames=[],
        kwargs={"dataset": "t0", "train_dir": output_path, "model": "google/t5-xl-lm-adapt"}
    )
    # this will create all the required data in the correct path
    T0PretrainDataModule(t0_config).setup()


if __name__ == '__main__':
    main()
