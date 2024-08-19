import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from mttl.config import ExpertConfig, MoEExpertConfig, MultiExpertConfig
from mttl.dataloader.flan_utils import download_flan
from mttl.datamodule.mt_seq_to_seq_module import FlanConfig, FlanModule
from mttl.models.expert_model import MultiExpertModel
from mttl.models.library.expert import Expert
from mttl.models.library.expert_library import DatasetLibrary
from mttl.models.modifiers.base import ModifierConfig
from mttl.models.modifiers.lora import LoRAConfig


@pytest.fixture(scope="session")
def gpt_neo():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")


@pytest.fixture(scope="session")
def tiny_flan(tmp_path_factory):
    """Prepare dataset for testing (once per session). Returns the dataset and the dataset_id.
    Downloads the first 100_000 examples (from a total of 23_173_509) and limit each task to 100 examples.
    The resulting dataset consists of 1800 examples from 18 tasks.
    tasks: {
        'cot_creak', 'cot_creak_ii', 'cot_ecqa', 'cot_ecqa_ii',
        'cot_esnli', 'cot_esnli_ii', 'cot_gsm8k', 'cot_gsm8k_ii',
        'cot_qasc', 'cot_qasc_ii', 'cot_sensemaking',
        'cot_sensemaking_ii', 'cot_strategyqa', 'cot_strategyqa_ii',
        'stream_aqua', 'stream_aqua_ii', 'stream_qed', 'stream_qed_ii'
    }
    template_type: {'fs_opt', 'zs_opt'}
    task_source: {'CoT'}
    """
    dataset = download_flan(
        split="train", download_size=100000, cutoff=100, verbose=False
    )
    dataset_path = tmp_path_factory.mktemp("flan-test-fixture")
    dataset_id = f"local://{dataset_path}"
    DatasetLibrary.push_dataset(dataset, dataset_id)
    return dataset, dataset_id


@pytest.fixture(scope="session")
def tiny_flan_dataset(tiny_flan):
    """Returns only the tiny flan dataset"""
    dataset, _ = tiny_flan
    return dataset


@pytest.fixture(scope="session")
def tiny_flan_id(tiny_flan):
    """Returns only the tiny flan dataset id"""
    _, dataset_id = tiny_flan
    return dataset_id


@pytest.fixture(scope="session")
def flan_data_module(tiny_flan_id):
    flan_dm = FlanModule(
        FlanConfig(
            dataset=tiny_flan_id,
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            max_input_length=1024,
            max_output_length=128,
            train_batch_size=4,
            predict_batch_size=1,
            truncation_side="left",
            include_template_type="zs_opt",
            include_task_source="CoT",
        ),
        for_generation=True,
        val_mixin=False,
    )
    return flan_dm


@pytest.fixture
def make_tiny_llama():
    def _tiny_llama():
        small_config = LlamaConfig(
            vocab_size=400,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=5,
            num_attention_heads=8,
            max_position_embeddings=512,
        )

        model_object = LlamaForCausalLM(small_config)
        return model_object

    return _tiny_llama


@pytest.fixture
def tiny_llama(make_tiny_llama):
    return make_tiny_llama()


@pytest.fixture
def tmp_lora_config(tmp_path: Path):
    return LoRAConfig(
        lora_rank=16,
        lora_alpha=1.0,
        modify_layers="c_fc|c_proj|k_proj|v_proj|q_proj|out_proj",
        modify_modules=".*",
    )


@pytest.fixture
def tmp_exp_config(tmp_path: Path):
    return ExpertConfig(
        model="EleutherAI/gpt-neo-125m",
        library_id=None,
        lora_rank=16,
        lora_alpha=1.0,
        model_modifier="lora",
        modify_layers="c_fc|c_proj|k_proj|v_proj|q_proj|out_proj",
        modify_modules=".*",
        trainable_param_names=".*lora_[ab].*",
        output_dir=tmp_path,
    )


@pytest.fixture
def tmp_multi_exp_config(tmp_path: Path):
    return MultiExpertConfig(
        model="EleutherAI/gpt-neo-125m",
        library_id=None,
        lora_rank=16,
        lora_alpha=1.0,
        model_modifier="lora",
        modify_layers="c_fc|c_proj|k_proj|v_proj|q_proj|out_proj",
        modify_modules=".*",
        trainable_param_names=".*lora_[ab].*",
        output_dir=tmp_path,
        router_selector="poly_router_dir",
        router_granularity="coarsegrained",
        n_tasks=1,
    )


@pytest.fixture
def tmp_peer_moe_config(tmp_path: Path):
    return MoEExpertConfig(
        model="EleutherAI/gpt-neo-125m",
        library_id=None,
        model_modifier="peer",
        modify_layers="",
        top_k=2,
        moe_num_experts=100,
        modify_modules=".*mlp",
        trainable_param_names=".*mlp.*",
        output_dir=tmp_path,
        router_selector="moe_pk_router",
        pk_use_batchnorm=False
    )


@pytest.fixture
def tmp_moe_exp_config(tmp_path):
    from mttl.config import MoEExpertConfig

    return MoEExpertConfig(
        model="EleutherAI/gpt-neo-125m",
        library_id=None,
        lora_rank=16,
        lora_alpha=1.0,
        model_modifier="lora",
        modify_layers="c_fc|c_proj|k_proj|v_proj|q_proj|out_proj",
        modify_modules=".*",
        trainable_param_names=".*lora_[ab].*",
        output_dir=tmp_path,
        router_selector="poly_router_dir",
        router_granularity="coarsegrained",
        n_tasks=1,
    )


@pytest.fixture
def create_dummy_expert(make_tiny_llama):
    def _create_dummy_expert(config: MultiExpertConfig, exp_name, **kwargs) -> Expert:
        if "model_object" not in kwargs and (
            config.model is None or config.model == ""
        ):
            # use tiny llama by default
            kwargs["model_object"] = make_tiny_llama()

        model = MultiExpertModel(**config.asdict(), **kwargs)
        expert = model.add_empty_expert(
            exp_name, ModifierConfig.from_training_config(config)
        )
        return expert

    return _create_dummy_expert


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
