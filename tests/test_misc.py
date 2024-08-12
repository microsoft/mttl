from mttl.cli.dataset_create import download_flan


def test_download_slice():
    dataset = download_flan(split="train", download_size=20)
    assert len(dataset) == 20


def test_download_slice_cutoff():
    dataset = download_flan(split="train", download_size=20, cutoff=1)
    assert len(dataset) == 9  # First 20 examples have 9 unique tasks
