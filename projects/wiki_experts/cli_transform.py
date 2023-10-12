import sys
import os
import click

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.utils import setup_logging


@click.group()
def cli():
    setup_logging()


@cli.command("transform")
@click.option(
    "--seed-dataset", type=str, default="sordonia/my-wiki-latex_mmlu_from_valid_all"
)
@click.option("--subjects", type=str, default="SUB_10")
@click.option("--config", type=str, required=True)
@click.option("--output_path", type=str, required=False, default="./")
@click.option("--upload_to_hub", type=bool, required=False, default=False)
def transform(
    seed_dataset,
    subjects,
    config,
    output_path,
    upload_to_hub=False,
):
    from src.data_transforms.config import AutoConfig
    from src.data_transforms.data_transforms import AutoTransform

    config = AutoConfig.from_path(config)
    transform = AutoTransform.from_config(config)

    output_path = os.environ.get("AMLT_OUTPUT_DIR", output_path)
    if upload_to_hub:
        assert (
            os.environ.get("HF_TOKEN") is not None
        ), "Please set HF_TOKEN env variable."

    transform.transform(
        seed_dataset,
        filter_subjects=subjects,
        num_iterations=config.num_iterations,
        output_path=output_path,
        upload_to_hub=upload_to_hub,
    )


@cli.command("upload")
@click.option("--dataset-path", type=str, required=True)
@click.option("--hf-destination", type=str, required=False, default=None)
def upload_to_hf(dataset_path, hf_destination=None):
    from src.data_transforms.utils import upload_to_hf_

    return upload_to_hf_(dataset_path, hf_destination=hf_destination)


if __name__ == "__main__":
    cli()
