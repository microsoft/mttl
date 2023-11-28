# unit test for adapter_ranker
import pytest
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary


def test_expert_lib(mocker):
    library = HFExpertLibrary("sordonia/test-library")
    assert len(library) == 20
    assert not library._sliced

    module_name = list(library.keys())[0]
    module_dump = library[module_name]

    library._upload_metadata = mocker.MagicMock()
    library._upload_weights = mocker.MagicMock()
    library._update_readme = mocker.MagicMock()

    # expert already there
    with pytest.raises(ValueError):
        library.add_expert(module_name, module_dump)

    assert module_dump.expert_config.model == "phi-2"
    assert len(module_dump.expert_weights) == 128
    assert module_dump.expert_info.parent_node is None
    assert module_dump.expert_info.expert_name is None

    library.add_expert("new_module", module_dump)
    assert library._upload_metadata.call_count == 1
    assert library._upload_weights.call_count == 1
    assert library._update_readme.call_count == 1
    assert len(library) == 21

    library = HFExpertLibrary(
        "sordonia/test-library", model_name="EleutherAI/other-model"
    )
    assert len(library) == 0
    assert library._sliced
