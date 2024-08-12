import asyncio
import io
import os
import uuid
from unittest.mock import patch

import pytest
import torch
from datasets import load_dataset
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
)

from mttl.models.library.expert_library import (
    BlobExpertLibrary,
    BlobStorageEngine,
    DatasetLibrary,
    ExpertLibrary,
    HFExpertLibrary,
    LocalExpertLibrary,
    LocalFSEngine,
    VirtualLocalLibrary,
)


def test_expert_lib(mocker):
    library = HFExpertLibrary("sordonia/new-test-library")
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
        "sordonia/new-test-library", model_name="EleutherAI/other-model"
    )
    assert len(library) == 0
    assert library._sliced

    library = HFExpertLibrary(
        "sordonia/new-test-library", exclude_selection=["abstract_algebra"]
    )

    assert len(library) == 1
    assert library._sliced

    with pytest.raises(ValueError):
        module_dump = library["abstract_algebra"]


def test_soft_delete(mocker):
    from mttl.models.library.expert_library import HFExpertLibrary

    # read the stored embeddings
    library = HFExpertLibrary("sordonia/new-test-library", create=False)
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


def test_read_embeddings():
    from mttl.models.library.expert_library import HFExpertLibrary

    # read the stored embeddings
    embeddings = HFExpertLibrary("sordonia/new-test-library").get_auxiliary_data(
        "embeddings"
    )
    assert "abstract_algebra" in embeddings
    assert embeddings["abstract_algebra"].shape[0] == 2


def test_add_auxiliary_data(mocker, tmp_path):
    from mttl.models.library.expert_library import HFExpertLibrary, LocalExpertLibrary

    library = LocalExpertLibrary.from_expert_library(
        HFExpertLibrary("sordonia/new-test-library"), tmp_path
    )

    library.add_auxiliary_data(
        data_type="test",
        expert_name="abstract_algebra",
        config={"name": "test_expert"},
        data={"test": 1},
    )
    assert (
        library.get_auxiliary_data("test", expert_name="abstract_algebra")["test"] == 1
    )
    assert (
        library.get_auxiliary_data(
            "test", expert_name="abstract_algebra", return_config=True
        )[0]["name"]
        == "test_expert"
    )


token = os.getenv("BLOB_SAS_TOKEN")


@pytest.fixture
def build_local_files():
    def _build_files(local_path, num_files):
        filenames = []
        for i in range(1, num_files + 1):
            blob_file_name = f"blob_data_{i}.txt"
            filenames.append(blob_file_name)
            local_data_path = local_path / blob_file_name
            with open(local_data_path, "wb") as my_blob:
                my_blob.write(f"Blob data {i}".encode())
        return filenames

    return _build_files


@pytest.fixture
def repo_id():
    return f"mttldata/{uuid.uuid4()}"


@pytest.fixture
def setup_repo():
    """Create a repository and delete it once the test is done"""
    engine_repo_refs = []

    def _create_repo(engine, repo_id, repo_type="models", exist_ok=True):
        engine_repo_refs.append((engine, repo_id))
        engine.create_repo(repo_id, repo_type=repo_type, exist_ok=exist_ok)
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
def test_async_upload_and_delete_blobs(
    tmp_path, build_local_files, setup_repo, repo_id
):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    # Write two temp files to upload
    local_path = engine.get_repository_cache_dir(repo_id)
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
    local_path = engine.get_repository_cache_dir(repo_id)
    filenames = build_local_files(local_path, 2)
    _ = asyncio.run(engine.async_upload_blobs(repo_id, filenames))

    # Remove cached files
    for filename in filenames:
        (local_path / filename).unlink()
    assert os.listdir(local_path) == []

    # Download the files from the remote
    blob_data_path = engine.snapshot_download(repo_id)

    assert blob_data_path == str(local_path)
    assert set(os.listdir(blob_data_path)) == {f"blob_data_1.txt", f"blob_data_2.txt"}


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
@pytest.mark.parametrize(
    "allow_patterns,expected_files",
    [
        (["blob_data_1.txt"], ["blob_data_1.txt"]),
        (["*.txt"], ["blob_data_1.txt", "blob_data_2.txt"]),
        ("*.txt", ["blob_data_1.txt", "blob_data_2.txt"]),  # str is also allowed
        (["blob_data_1.*"], ["blob_data_1.txt", "blob_data_1.json"]),
        (["*1.txt", "*2.json"], ["blob_data_1.txt", "blob_data_2.json"]),
        (
            None,
            [
                "blob_data_1.txt",
                "blob_data_2.txt",
                "blob_data_1.json",
                "blob_data_2.json",
            ],
        ),
        ([], []),
        ("", []),
    ],
)
def test_snapshot_download_filtered(
    tmp_path, setup_repo, repo_id, allow_patterns, expected_files
):
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)

    # Upload two temp files to download
    local_path = engine.get_repository_cache_dir(repo_id)
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

    local_path = engine.get_repository_cache_dir(repo_id)
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

    local_path = engine.get_repository_cache_dir(repo_id)
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

    local_path = engine.get_repository_cache_dir(repo_id)
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


@pytest.fixture
def build_meta_ckpt():
    def _build_meta_ckpt(local_path, num_files):
        readme = "README.md"
        readme_filepath = local_path / readme
        readme_filepath.touch()
        filenames = [readme]
        for i in range(1, num_files + 1):
            expert_name = f"expert_{i}"
            metadata = f"{expert_name}.meta"
            ckpt = f"{expert_name}.ckpt"
            filenames.extend([metadata, ckpt])

            local_meta_path = local_path / metadata
            metadata_data = {
                "expert_name": expert_name,
                "expert_deleted": False,
                "model_name": f"model_{i}",
                "training_config": {
                    "model": f"model_{i}",
                },
            }
            torch.save(metadata_data, local_meta_path)

            ckpt_filepath = local_path / ckpt
            dummy_state = {"state_dict": {}}
            torch.save(dummy_state, ckpt_filepath)

        return filenames

    return _build_meta_ckpt


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_copy_library_blob_to_blob(tmp_path, build_meta_ckpt, setup_repo, repo_id):
    # Create a library with two experts
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)
    local_path = engine.get_repository_cache_dir(repo_id)
    filenames = build_meta_ckpt(local_path, 2)
    _ = asyncio.run(engine.async_upload_blobs(repo_id, filenames))

    # Get the expert library
    az_repo_id = f"az://{repo_id}"
    library = ExpertLibrary.get_expert_library(az_repo_id)

    # Create a new library from the first one
    new_repo_id = f"local://mttldata/{uuid.uuid4()}"
    az_new_repo_id = f"az://{new_repo_id}"
    try:  # Clean up the new repo if the test fails
        new_lib = BlobExpertLibrary.from_expert_library(library, az_new_repo_id)
        assert set(new_lib.list_repo_files(new_repo_id)) == set(
            library.list_repo_files(repo_id)
        )
    finally:
        BlobExpertLibrary(az_new_repo_id).delete_repo(new_repo_id)


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_copy_library_blob_to_local(tmp_path, build_meta_ckpt, setup_repo, repo_id):
    # Create a library with two experts
    engine = BlobStorageEngine(token=token, cache_dir=tmp_path)
    setup_repo(engine, repo_id)
    local_path = engine.get_repository_cache_dir(repo_id)
    filenames = build_meta_ckpt(local_path, 2)
    _ = asyncio.run(engine.async_upload_blobs(repo_id, filenames))

    # Get the expert library
    az_repo_id = f"az://{repo_id}"
    library = ExpertLibrary.get_expert_library(az_repo_id)

    # Create a new library from the first one
    new_repo_path = tmp_path / "new_repo"
    new_repo_path.mkdir()
    new_repo_id = f"local://{new_repo_path}"
    new_lib = LocalExpertLibrary.from_expert_library(library, new_repo_id)
    # drop the path and keep the filenames
    local_files = {f.split("/")[-1] for f in new_lib.list_repo_files(new_repo_id)}
    assert local_files == set(library.list_repo_files(repo_id))


def test_copy_library_local_to_local(tmp_path, build_meta_ckpt, setup_repo, repo_id):
    # Create a library with two experts
    local_path = tmp_path / "base_repo"
    local_path.mkdir()
    engine = LocalFSEngine()
    setup_repo(engine, local_path)
    filenames = build_meta_ckpt(local_path, 2)
    repo_id = f"local://{local_path}"

    # Get the expert library
    library = ExpertLibrary.get_expert_library(repo_id)

    # Create a new library from the first one
    new_repo_path = tmp_path / "new_repo"
    new_repo_path.mkdir()
    new_repo_id = f"local://{new_repo_path}"
    new_lib = LocalExpertLibrary.from_expert_library(library, new_repo_id)
    # drop the path and keep the filenames
    base_files = {f.split("/")[-1] for f in library.list_repo_files(repo_id)}
    new_files = {f.split("/")[-1] for f in new_lib.list_repo_files(new_repo_id)}
    assert base_files == new_files


def test_get_expert_library_copy(tmp_path, build_meta_ckpt, setup_repo, repo_id):
    # Create a library with two experts
    local_path = tmp_path / "base_repo"
    local_path.mkdir()
    engine = LocalFSEngine()
    setup_repo(engine, local_path)
    filenames = build_meta_ckpt(local_path, 2)
    repo_id = f"local://{local_path}"

    # Get the expert library
    library = ExpertLibrary.get_expert_library(repo_id)

    # Get the expert library creating a copy of the original
    new_repo_id = f"local://{tmp_path / 'new_repo'}"
    new_lib = ExpertLibrary.get_expert_library(repo_id, destination_id=new_repo_id)

    # drop the path and keep the filenames
    base_files = {f.split("/")[-1] for f in library.list_repo_files(repo_id)}
    new_files = {f.split("/")[-1] for f in new_lib.list_repo_files(new_repo_id)}
    assert base_files == new_files


def test_virtual_library_is_in_memory(tmp_path, build_meta_ckpt, setup_repo, repo_id):
    # Create a library with two experts
    local_path = tmp_path / "base_repo"
    local_path.mkdir()
    engine = LocalFSEngine()
    setup_repo(engine, local_path)
    filenames = build_meta_ckpt(local_path, 2)
    repo_id = f"local://{local_path}"
    # Get the expert library
    local_library = ExpertLibrary.get_expert_library(repo_id)

    # Create a new library from the first one
    virtual_lib = VirtualLocalLibrary.from_expert_library(
        local_library, local_library.repo_id
    )
    assert set(virtual_lib.data.keys()) == {"expert_1", "expert_2"}

    virtual_lib.remove_expert("expert_2")
    assert set(virtual_lib.data.keys()) == {"expert_1"}

    # check that the original library is not affected
    base_files = {f.split("/")[-1] for f in local_library.list_repo_files(local_path)}
    assert base_files == {
        "README.md",
        "expert_1.meta",
        "expert_1.ckpt",
        "expert_2.meta",
        "expert_2.ckpt",
    }
    assert set(os.listdir(local_path)) == {
        "README.md",
        "expert_1.meta",
        "expert_1.ckpt",
        "expert_2.meta",
        "expert_2.ckpt",
    }


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
@pytest.mark.parametrize(
    "expert_lib_class, repo_id",
    [
        (BlobExpertLibrary, "az://mttldata/abc"),
        (HFExpertLibrary, "hf://mttldata/abc"),
        (LocalExpertLibrary, "local://mttldata/abc"),
        (VirtualLocalLibrary, "virtual://mttldata/abc"),
        (LocalExpertLibrary, "mttldata/abc"),
    ],
)
def test_get_expert_library(expert_lib_class, repo_id):
    class_name = expert_lib_class.__name__
    with patch(f"mttl.models.library.expert_library.{class_name}") as mock_expert_lib:
        expert_library = ExpertLibrary.get_expert_library(repo_id)
        mock_expert_lib.assert_called_once()
        assert class_name in str(expert_library)


def test_dataset_library(tmp_path, tiny_flan_dataset):
    token = None  # not needed for local, just demostrating the API
    dataset_path = tmp_path / "flan-test"
    dataset_id = f"local://{dataset_path}"
    # dataset_id = "hf://<hf-user-name>/flan-test"
    # dataset_id = f"az://<storage-account>/flan-test"
    try:
        DatasetLibrary.push_dataset(tiny_flan_dataset, dataset_id, token=token)
        dataset_from_lib = DatasetLibrary.pull_dataset(dataset_id, token=token)
        assert tiny_flan_dataset.data == dataset_from_lib["train"].data
    finally:  # Clean up. Delete the dataset from the library
        DatasetLibrary.delete_dataset(dataset_id, token=token)


def test_dataset_library_kwargs(tmp_path):
    dataset_id = "winogrande"
    config_name = "winogrande_xs"
    split = "train"

    # load from huggingface
    dataset = load_dataset(dataset_id, name=config_name)

    # load dataset with config name
    dataset_from_lib = DatasetLibrary.pull_dataset(dataset_id, name=config_name)
    assert dataset.keys() == dataset_from_lib.keys()
    assert dataset[split].data == dataset_from_lib[split].data

    # filter with split
    dataset_from_lib = DatasetLibrary.pull_dataset(
        dataset_id, name=config_name, split=split
    )
    assert dataset[split].data == dataset_from_lib.data

    # save and load from local with name and split
    local_path = tmp_path / dataset_id
    local_dataset_id = f"local://{local_path}"
    DatasetLibrary.push_dataset(
        dataset, local_dataset_id, name=config_name, token=token
    )
    local_dataset_from_lib = DatasetLibrary.pull_dataset(
        local_dataset_id, name=config_name, token=token
    )
    assert dataset.keys() == local_dataset_from_lib.keys()
    assert dataset[split].data == local_dataset_from_lib[split].data

    # load a split
    local_dataset_from_lib = DatasetLibrary.pull_dataset(
        local_dataset_id, name=config_name, split=split, token=token
    )
    assert dataset[split].data == local_dataset_from_lib.data


@pytest.mark.skipif(token is None, reason="Requires access to Azure Blob Storage")
def test_dataset_library_kwargs(tmp_path):
    dataset_id = "winogrande"
    config_name = "winogrande_xs"
    split = "train"

    # load from huggingface
    dataset = load_dataset(dataset_id, name=config_name)

    # save and load from local with name and split
    local_path = tmp_path / dataset_id
    blob_dataset_id = f"az://mttldata/{dataset_id}"
    try:
        DatasetLibrary.push_dataset(
            dataset, blob_dataset_id, name=config_name, token=token
        )
        dataset_from_blob = DatasetLibrary.pull_dataset(
            blob_dataset_id, name=config_name, token=token
        )
        assert dataset.keys() == dataset_from_blob.keys()
        assert dataset[split].data == dataset_from_blob[split].data

        # load a split
        dataset_from_blob = DatasetLibrary.pull_dataset(
            blob_dataset_id, name=config_name, split=split, token=token
        )
        assert dataset[split].data == dataset_from_blob.data

    finally:  # Clean up. Delete the dataset from the library
        DatasetLibrary.delete_dataset(blob_dataset_id, token=token)
