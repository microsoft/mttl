# unit test for adapter_ranker
import pytest
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from projects.wiki_experts.src.expert_library import HFExpertLibrary
from projects.wiki_experts.src.config import ExpertConfig


def test_expert_lib():
    library = HFExpertLibrary(
        "sordonia/test-library-for-neo-125m", "EleutherAI/gpt-neo-125m"
    )
    assert len(library) == 1

    # expert already there
    with pytest.raises(ValueError):
        library.add_expert(list(library.keys())[0], list(library.values())[0])

    # wrong model
    with pytest.raises(ValueError):
        config = list(library.values())[0].expert_config
        config.model = "my-other-model"
        library.add_expert("new_adapter", list(library.values())[0])
