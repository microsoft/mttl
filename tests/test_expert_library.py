# unit test for adapter_ranker
import io
import os
import pytest
import asyncio
import uuid
from mttl.models.modifiers.expert_containers.expert_library import (
    BlobStorageEngine,
    HFExpertLibrary,
)

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    CommitOperationCopy,
)


def test_expert_lib(mocker):
    library = HFExpertLibrary("sordonia/test-library")
    assert len(library) == 2
    assert not library._sliced

    module_dump = library["abstract_algebra"]

    library._upload_metadata = mocker.MagicMock()
    library._upload_weights = mocker.MagicMock()
    library._update_readme = mocker.MagicMock()

    # expert already there
    with pytest.raises(ValueError):
        library.add_expert(module_dump, "abstract_algebra")

    assert module_dump.expert_info.model == "phi-2"
    assert len(module_dump.expert_weights) == 128
    assert module_dump.expert_info.parent_node is None
    assert module_dump.expert_info.expert_name == "abstract_algebra"

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

    library = HFExpertLibrary(
        "sordonia/test-library", exclude_selection=["abstract_algebra"]
    )

    assert len(library) == 1
    assert library._sliced

    with pytest.raises(ValueError):
        module_dump = library["abstract_algebra"]


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

    # TODO: both work, decide which one to use
    library = HFExpertLibrary("sordonia/test-library")
    # library = get_expert_library("test-library")  # requires BLOB_SAS_URL env var
    embeddings, svd = SVDEmbeddingTransform(
        SVDEmbeddingTransformConfig(n_components=2)
    ).transform(library=library, upload_to_hf=False)
    assert embeddings.shape[1] == 2


def test_read_embeddings():
    from mttl.models.modifiers.expert_containers.expert_library import HFExpertLibrary

    # read the stored embeddings
    embeddings = HFExpertLibrary("sordonia/test-library").get_auxiliary_data(
        "embeddings"
    )
    assert "abstract_algebra" in embeddings
    assert embeddings["abstract_algebra"]["svd"]["embedding"].shape[1] == 2


def test_add_auxiliary_data(mocker, tmp_path):
    from mttl.models.modifiers.expert_containers.expert_library import (
        HFExpertLibrary,
        LocalExpertLibrary,
    )

    # read the stored embeddings
    library = LocalExpertLibrary.create_from_remote(
        HFExpertLibrary("sordonia/test-library"), tmp_path
    )

    library.add_auxiliary_data(
        data_type="test",
        expert_name="abstract_algebra",
        config={"name": "test_expert"},
        data={"test": 1},
    )
    assert (
        library.get_auxiliary_data("test", expert_name="abstract_algebra")[
            "test_expert"
        ]["test"]["test"]
        == 1
    )


token = os.getenv("BLOB_SAS_URL")

@pytest.fixture
def build_local_files():
    def _build_files(local_path, num_files):
        filenames = []
        for i in range(1, num_files + 1):
            blob_file_name = f"blob_data_{i}.txt"
            filenames.append(blob_file_name)
            local_data_path = str(local_path / blob_file_name)
            with open(local_data_path, "wb") as my_blob:
                my_blob.write(f"Blob data {i}".encode())
        return filenames
    return _build_files

@pytest.fixture
def repo_id():
    return str(uuid.uuid4())

@pytest.fixture
def setup_repo():
    """Create a repository and delete it once the test is done"""
    engine_repo_refs = []
    def _create_repo(engine, repo_id):
        engine_repo_refs.append((engine, repo_id))
        engine.create_repo(repo_id)
        files = engine.list_repo_files(repo_id)
        assert not files
    yield _create_repo
    [e.delete_repo(r) for e, r in engine_repo_refs]


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_create_and_delete_repo(tmp_path, repo_id):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    engine.delete_repo(repo_id)
    engine.create_repo(repo_id)
    engine.create_repo(repo_id, exist_ok=True)
    with pytest.raises(ValueError):
        engine.create_repo(repo_id, exist_ok=False)
    engine.delete_repo(repo_id)


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_async_upload_and_delete_blobs(tmp_path, build_local_files, setup_repo, repo_id):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    # Write two temp files to upload
    local_path = tmp_path / repo_id
    local_path.mkdir()
    filenames = build_local_files(local_path, 2)
    _ = asyncio.run(engine.async_upload_blobs(repo_id, filenames))
    files = engine.list_repo_files(repo_id)
    assert set(files) == {"blob_data_1.txt", "blob_data_2.txt"}


    asyncio.run(engine.async_delete_blobs(repo_id, filenames))
    files = engine.list_repo_files(repo_id)
    assert not files


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_snapshot_download(tmp_path, build_local_files, setup_repo, repo_id):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    # Upload two temp files to download
    local_path = tmp_path / repo_id
    local_path.mkdir()
    filenames = build_local_files(local_path, 2)
    _ = asyncio.run(engine.async_upload_blobs(repo_id, filenames))

    # Remove cached files
    for filename in filenames:
        (local_path / filename).unlink()
    assert os.listdir(local_path) == []

    # Download the files from the remote
    blob_data_path = engine.snapshot_download(repo_id)

    assert blob_data_path == str(local_path)
    assert set(os.listdir(blob_data_path)) == {
        f"blob_data_1.txt",
        f"blob_data_2.txt"
    }


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
@pytest.mark.parametrize("allow_patterns,expected_files", [
    (["blob_data_1.txt"], ["blob_data_1.txt"]),
    (["*.txt"], ["blob_data_1.txt", "blob_data_2.txt"]),
    ("*.txt", ["blob_data_1.txt", "blob_data_2.txt"]),  # str is also allowed
    (["blob_data_1.*"], ["blob_data_1.txt", "blob_data_1.json"]),
    (["*1.txt", "*2.json"], ["blob_data_1.txt", "blob_data_2.json"]),
    (None, ["blob_data_1.txt", "blob_data_2.txt", "blob_data_1.json", "blob_data_2.json"]),
    ([], []),
    ("", []),
])
def test_snapshot_download_filtered(tmp_path, setup_repo, repo_id, allow_patterns, expected_files):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    # Upload two temp files to download
    local_path = tmp_path / repo_id
    local_path.mkdir()
    filenames = [
        "blob_data_1.txt",
        "blob_data_2.txt",
        "blob_data_1.json",
        "blob_data_2.json",
    ]
    for filename in filenames:
        local_data_path = str(local_path / filename)
        with open(local_data_path, "wb") as my_blob:
            my_blob.write(f"{filename}".encode())
    _ = asyncio.run(engine.async_upload_blobs(repo_id, filenames))

    # Remove cached files
    for filename in filenames:
        (local_path / filename).unlink()
    assert os.listdir(local_path) == []

    # Download the files from the remote
    blob_data_path = engine.snapshot_download(repo_id, allow_patterns=allow_patterns)

    assert blob_data_path == str(local_path)
    assert set(os.listdir(blob_data_path)) == set(expected_files)


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_hf_hub_download(tmp_path, build_local_files, setup_repo, repo_id):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    local_path = tmp_path / repo_id
    local_path.mkdir()
    filenames = build_local_files(local_path, 2)
    _ = asyncio.run(engine.async_upload_blobs(repo_id, filenames))

    for filename in filenames:
        (local_path / filename).unlink()
    assert os.listdir(local_path) == []

    filename_1, filename_2 = filenames

    # Download the file 1 from the remote
    blob_data_path = engine.hf_hub_download(repo_id, filename_1)
    assert blob_data_path == str(local_path / filename_1)
    with open(blob_data_path, "rb") as f:
        assert f.read() == b"Blob data 1"
    assert os.listdir(local_path) == [filename_1]

    # Two files stored in the remote repo
    repo_files = engine.list_repo_files(repo_id)
    assert set(repo_files) == {filename_1, filename_2}


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_create_commit_sync(tmp_path, build_local_files, setup_repo, repo_id):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    local_path = tmp_path / repo_id
    local_path.mkdir()
    filenames = build_local_files(local_path, 4)
    f1, f2, f3, f4 = filenames
    f5 = "blob_data_5.txt"
    f6 = "blob_data_6.txt"

    # Upload f3 and f4
    _ = asyncio.run(engine.async_upload_blobs(repo_id, [f3, f4]))

    # Load f1 into memory, keep f2 as a file object
    with open(local_path / f1, "rb") as f:
        buffer = io.BytesIO(f.read())

    # Create ops: upload f1 and f2, copy f3 to f5 and f4 to f5, delete f3 and f5
    ops = [
        CommitOperationAdd(path_in_repo=f1, path_or_fileobj=buffer),
        CommitOperationAdd(path_in_repo=f2, path_or_fileobj=local_path / f2),
        CommitOperationCopy(src_path_in_repo=f3, path_in_repo=f5),
        CommitOperationCopy(src_path_in_repo=f4, path_in_repo=f6),
        CommitOperationDelete(path_in_repo=f3),
        CommitOperationDelete(path_in_repo=f5),
    ]

    engine.create_commit(repo_id, ops, "Commit operations in order")

    assert set(engine.list_repo_files(repo_id)) == {f1, f2, f4, f6}
    f6_path = engine.hf_hub_download(repo_id, f6)
    with open(f6_path, "rb") as f:
        assert f.read() == b"Blob data 4"


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_create_commit_async(tmp_path, build_local_files, setup_repo, repo_id):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    local_path = tmp_path / repo_id
    local_path.mkdir()
    filenames = build_local_files(local_path, 4)
    f1, f2, f3, f4 = filenames

    ops = [
        CommitOperationAdd(path_in_repo=f1, path_or_fileobj=local_path / f1),
        CommitOperationAdd(path_in_repo=f2, path_or_fileobj=local_path / f2),
        CommitOperationAdd(path_in_repo=f3, path_or_fileobj=local_path / f3),
        CommitOperationAdd(path_in_repo=f4, path_or_fileobj=local_path / f4),
    ]

    engine.create_commit(repo_id, ops, "Push files async", async_mode=True)

    assert set(engine.list_repo_files(repo_id)) == {f1, f2, f3, f4}

    ops = [
        CommitOperationDelete(path_in_repo=f1),
        CommitOperationDelete(path_in_repo=f2),
        CommitOperationDelete(path_in_repo=f3),
        CommitOperationDelete(path_in_repo=f4),
    ]

    engine.create_commit(repo_id, ops, "Delete files async", async_mode=True)
    assert not engine.list_repo_files(repo_id)