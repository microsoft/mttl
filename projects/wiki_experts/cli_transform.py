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
@click.option("--subjects", type=str, default="ALL")
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
    from src.data_transforms.base import TransformConfig
    from src.data_transforms.base import TransformModel

    config = TransformConfig.from_path(config)
    transform = TransformModel.from_config(config)

    output_path = os.environ.get("AMLT_OUTPUT_DIR", output_path)
    if upload_to_hub:
        assert (
            os.environ.get("HF_TOKEN") is not None
        ), "Please set HF_TOKEN env variable."

    transform.transform(
        seed_dataset,
        filter_subjects=subjects,
        output_path=output_path,
        upload_to_hub=upload_to_hub,
    )


@cli.command("upload")
@click.option("--dataset-path", type=str, required=True)
@click.option("--config-file", type=str, required=False)
@click.option("--flat", type=bool, default=True)
@click.option("--create-split", type=bool, default=False)
@click.option("--aug_few_shot", type=int, required=False, default=-1)
@click.option("--hf-destination", type=str, required=False, default=None)
def upload_to_hf(
    dataset_path,
    hf_destination=None,
    config_file=None,
    flat=True,
    create_split=False,
    aug_few_shot=-1,
):
    from src.data_transforms.utils import upload_to_hf_

    if config_file is not None:
        from src.data_transforms.base import TransformConfig

        config = TransformConfig.from_path(config_file)

    return upload_to_hf_(
        dataset_path,
        hf_destination=hf_destination,
        configuration=config,
        flat=flat,
        create_split=create_split,
        aug_few_shot=aug_few_shot,
    )


if __name__ == "__main__":
    cli()
