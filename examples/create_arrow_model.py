import os

import click

from mttl.models.containers.selectors import ArrowSelectorConfig
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.library.library_transforms import ArrowConfig, ArrowTransform


@click.command()
@click.option(
    "--experts", type=str, help="Comma separated list of expert repos in Huggingface."
)
@click.option(
    "--push-to-hub",
    type=str,
    help="Destination where to save the 'Arrowed' model with the MTTL library.",
)
def make_arrow(experts, push_to_hub):
    repos = experts.split(",")
    if not repos:
        raise ValueError("Nothing to be done! Please provide paths to experts.")

    temporary_id = "virtual://library"

    # create a temporary library
    library = ExpertLibrary.get_expert_library(temporary_id, create=True)
    for path in repos:
        library.add_expert_from_ckpt(path)

    # compute arrow prototypes and store them in the library
    arrow_config = ArrowConfig()
    transform = ArrowTransform(arrow_config)
    transform.transform(library, persist=True)

    # save arrowed model to HF, can be reloaded with MultiExpertModel.from_pretrained(push_to_hub)
    arrow_selector = ArrowSelectorConfig(
        top_k=2, selector_data_id=arrow_config.save_name
    )

    breakpoint()
    expert_model = MultiExpertModel.from_pretrained_library(
        library, selector_config=arrow_selector
    )
    expert_model.push_to_hub(push_to_hub)

    # alternatively we could also store the library in HF (for ex, if we want to swap selectors with the same experts)


if __name__ == "__main__":
    make_arrow()
