# unit test for adapter_ranker
import pytest
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary


def test_expert_lib():
    library = HFExpertLibrary("sordonia/test-library-for-neo-125m")
    assert len(library) == 1
    assert not library._sliced
    assert not library._modified

    # expert already there
    with pytest.raises(ValueError):
        library.add_expert(list(library.keys())[0], list(library.values())[0])

    assert (
        library["quail_description_context_question_answer_text"].expert_config.model
        == "EleutherAI/gpt-neo-125m"
    )
    assert (
        len(library["quail_description_context_question_answer_text"].expert_weights)
        == 72
    )
    assert (
        library[
            "quail_description_context_question_answer_text"
        ].expert_info.parent_node
        is None
    )
    assert (
        library[
            "quail_description_context_question_answer_text"
        ].expert_info.expert_name
        is None
    )

    library.add_expert("new-expert", list(library.values())[0])
    assert len(library) == 2
    assert library._modified

    library = HFExpertLibrary(
        "sordonia/test-library-for-neo-125m", model_name="EleutherAI/other-model"
    )
    assert len(library) == 0
    assert library._sliced
