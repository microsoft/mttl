# unit test for adapter_ranker
import pytest
from mttl.datamodule.mt_seq_to_seq_module import FlanModule, FlanConfig
from projects.wiki_experts.src.expert_library import HFExpertLibrary
from projects.wiki_experts.src.config import ExpertConfig


def test_expert_lib():
    library = HFExpertLibrary("sordonia/test-library-for-neo-125m")
    assert len(library) == 1
    assert not library._sliced
    assert not library._modified

    # expert already there
    with pytest.raises(ValueError):
        library.add_expert(list(library.keys())[0], list(library.values())[0])

    library.add_expert("new-expert", list(library.values())[0])
    assert len(library) == 2
    assert library._modified

    library = HFExpertLibrary(
        "sordonia/test-library-for-neo-125m", model_name="EleutherAI/other-model"
    )
    assert len(library) == 0
    assert library._sliced
