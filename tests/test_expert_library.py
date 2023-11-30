# unit test for adapter_ranker
import pytest
from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary


def test_expert_lib(mocker):
    library = HFExpertLibrary("sordonia/test-library")
    assert len(library) == 2
    assert not library._sliced

    module_name = list(library.keys())[0]
    module_dump = library[module_name]

    library._upload_metadata = mocker.MagicMock()
    library._upload_weights = mocker.MagicMock()
    library._update_readme = mocker.MagicMock()

    # expert already there
    with pytest.raises(ValueError):
        library.add_expert(module_dump, module_name)

    assert module_dump.expert_config.model == "phi-2"
    assert len(module_dump.expert_weights) == 128
    assert module_dump.expert_info.parent_node is None
    assert (
        module_dump.expert_info.expert_name
        == "adversarial_qa_dbert_answer_the_following_q"
    )

    library.add_expert(module_dump, "new_module")
    assert library._upload_metadata.call_count == 1
    assert library._upload_weights.call_count == 1
    assert library._update_readme.call_count == 1
    assert len(library) == 3

    library = HFExpertLibrary(
        "sordonia/test-library", model_name="EleutherAI/other-model"
    )
    assert len(library) == 0
    assert library._sliced


def test_soft_delete(mocker):
    from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary

    # read the stored embeddings
    library = HFExpertLibrary("sordonia/test-library", create=False)
    assert len(library.data) == 2

    key = list(library.keys())[0]

    library._upload_metadata = mocker.MagicMock()
    library._update_readme = mocker.MagicMock()
    library.remove_expert(key, soft_delete=True)
    assert len(library.data) == 1
    assert key not in library.data
    assert library._upload_metadata.call_count == 1
    assert library._update_readme.call_count == 1

    library.unremove_expert(key)
    assert len(library.data) == 2


def test_compute_embeddings():
    from mttl.models.modifiers.expert_containers.library_transforms import (
        SVDEmbeddingTransform,
        SVDEmbeddingTransformConfig,
    )

    embeddings = SVDEmbeddingTransform(
        SVDEmbeddingTransformConfig(n_components=2)
    ).transform("sordonia/test-library", upload_to_hf=False)
    assert embeddings.shape[1] == 2


def test_read_embeddings():
    from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary

    # read the stored embeddings
    embeddings = HFExpertLibrary("sordonia/test-library").get_auxiliary_data(
        "embeddings"
    )
    assert "abstract_algebra" in embeddings
    assert embeddings["abstract_algebra"]["svd"]["embedding"].shape[1] == 2
