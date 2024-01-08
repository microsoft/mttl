from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request
import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def gpt_neo():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")


@pytest.fixture(scope="session")
def flan_batch_for_generation():
    from mttl.datamodule.base import AutoDataModule

    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        max_input_length=1024,
        train_batch_size=4,
        predict_batch_size=2,
        truncation_side="left",
        for_generation=True,
        include_template_type="zs_noopt",
        include_task_source="P3,Flan2021",
    )
    dl = flan.val_dataloader()
    batch = next(iter(dl))
    return batch


@pytest.fixture(scope="session")
def flan_batch_for_training():
    from mttl.datamodule.base import AutoDataModule

    flan = AutoDataModule.create(
        "sordonia/flan-debug-flat",
        model="EleutherAI/gpt-neo-125m",
        model_family="gpt",
        max_input_length=1024,
        train_batch_size=4,
        predict_batch_size=2,
        truncation_side="left",
    )
    dl = flan.val_dataloader()
    batch = next(iter(dl))
    return batch


def make_tiny_llama():
    from transformers.models.llama.configuration_llama import LlamaConfig

    small_config = LlamaConfig(
        vocab_size=400,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=5,
        num_attention_heads=8,
        max_position_embeddings=512,
    )
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    model_object = LlamaForCausalLM(small_config)
    return model_object


def setup_mmlu(session):
    # setup_mmlu
    if "MMLU_DATA_DIR" not in os.environ:
        td = TemporaryDirectory()
        tmp_path = Path(td.name)
        del td

        os.mkdir(tmp_path)
        urllib.request.urlretrieve(
            "https://people.eecs.berkeley.edu/~hendrycks/data.tar",
            tmp_path / f"mmlu.tar",
        )

        print("Extracting data.tar to", tmp_path)

        os.system(f"tar -xf {tmp_path}/mmlu.tar -C {tmp_path}")
        os.environ["MMLU_DATA_DIR"] = os.path.join(str(tmp_path), "data")
        session.__MMLU_CACHE = str(tmp_path)
    else:
        session.__MMLU_CACHE = None


def setup_gptneo(session):
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")


def teardown_mmlu(session):
    # remove the cached images
    import shutil

    if session.__MMLU_CACHE is not None:
        shutil.rmtree(session.__MMLU_CACHE, ignore_errors=True)


def pytest_sessionstart(session):
    setup_mmlu(session)
    setup_gptneo(session)


def pytest_sessionfinish(session, exitstatus):
    teardown_mmlu(session)
