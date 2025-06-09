from mttl.models.library.expert_library import ExpertLibrary
from mttl.models.lightning.expert_module import MultiExpertModule
from mttl.arguments import FinetuneConfig
from mttl.datamodule.base import get_datamodule
from projects.modular_llm.finetune_experts import train_module


def create_library_from_peft(args):

    if (
        not args.experts
    ):  # something like this "lorahub/flan_t5_large-super_glue_wic,lorahub/flan_t5_large-wiki_qa_Jeopardy_style"
        raise ValueError("Please provide paths to experts.")
    repos = args.experts.split(",")
    if not repos:
        raise ValueError("Nothing to be done! Please provide paths to experts.")

    # create a temporary library
    library = ExpertLibrary.get_expert_library(
        "hf://zhan1993/abstention_library", create=True
    )
    with library.batched_commit():
        for path in repos:
            library.add_expert_from_ckpt(path)


def train_routing_with_peft_library(args):

    # create a temporary library
    library = ExpertLibrary.get_expert_library(args.library_id, create=True)
    module = MultiExpertModule(**vars(args)).to("cuda")
    module.add_experts_from_library(library=library)
    dm = get_datamodule(args)
    train_module(args, module, dm)


def train_lora_soup_with_peft(args):
    if (
        not args.experts
    ):  # something like this "lorahub/flan_t5_large-super_glue_wic,lorahub/flan_t5_large-wiki_qa_Jeopardy_style"
        raise ValueError("Please provide paths to experts.")
    repos = args.experts.split(",")
    if not repos:
        raise ValueError("Nothing to be done! Please provide paths to experts.")

    temporary_id = "virtual://library"

    # create a temporary library
    library = ExpertLibrary.get_expert_library(temporary_id, create=True)
    for path in repos:
        library.add_expert_from_ckpt(path)

    module = MultiExpertModule(**vars(args)).to("cuda")
    module.add_experts_from_library(library=library)

    dm = get_datamodule(args)
    train_module(args, module, dm)
    # alternatively we could also store the library in HF (for ex, if we want to swap selectors with the same experts)


if __name__ == "__main__":
    args = FinetuneConfig.parse()
    # create_library_from_peft(args)
    # train_lora_soup_with_peft(args)
    train_routing_with_peft_library(args)
