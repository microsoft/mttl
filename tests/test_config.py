import json
from dataclasses import dataclass

import pytest

from mttl.config import Args, ModifierArgs


@pytest.fixture
def SimpleArgs(tmp_path):
    @dataclass
    class SimpleConfig(Args):
        train_dir: str = "train_dir"
        optimizer: str = "adafactor"
        dataset: str = "t0"
        model: str = "t5-small"
        total_steps: str = 1000
        learning_rate: str = 1e-3
        output_dir: str = str(tmp_path / "output_dir")
        attn_implementation: str = None

    return SimpleConfig


@pytest.fixture
def InheritFromModifierArgs(tmp_path):
    @dataclass
    class TestConfig(ModifierArgs):
        train_dir: str = "train_dir"
        optimizer: str = "adafactor"
        dataset: str = "t0"
        model: str = "t5-small"
        total_steps: str = 1000
        learning_rate: str = 1e-3
        output_dir: str = str(tmp_path / "output_dir")
        attn_implementation: str = None

    return TestConfig


def test_config_dict_like(tmp_path, SimpleArgs):
    train_dir = str(tmp_path)
    optimizer = "adafactor"
    dataset = "t0"
    model = "t5-small"
    total_steps = 1000
    learning_rate = 1e-3
    config_dict = {
        "optimizer": optimizer,
        "dataset": dataset,
        "model": model,
        "train_dir": train_dir,
        "total_steps": total_steps,
        "learning_rate": learning_rate,
    }
    config = SimpleArgs(**config_dict)
    reconstructed_config = SimpleArgs.fromdict(json.loads(json.dumps(config.asdict())))
    assert optimizer in reconstructed_config.optimizer
    assert dataset in reconstructed_config.dataset
    assert model in reconstructed_config.model
    assert train_dir in reconstructed_config.train_dir
    assert total_steps == reconstructed_config.total_steps
    assert learning_rate == reconstructed_config.learning_rate


def test_config_was_override_from_kwargs(SimpleArgs):
    config = SimpleArgs(
        **{
            "optimizer": "adafactor",
            "dataset": "t0",
            "model": "t5-large",
        }
    )
    assert not config.was_overridden("train_dir")
    assert not config.was_overridden("optimizer")
    assert config.was_overridden("model")
    assert config.optimizer == "adafactor"


def test_config_to_json(InheritFromModifierArgs):
    config = InheritFromModifierArgs(
        **{
            "optimizer": "adafactor",
            "dataset": "t0",
            "model": "t5-large",
        }
    )
    data = config.to_json()
    data = kwargs = json.loads(data)
    assert data["optimizer"] == "adafactor"
    assert data["dataset"] == "t0"
    assert data["model"] == "t5-large"
    assert data["args_class"] == "TestConfig"


def test_config_was_override_from_file(tmp_path, SimpleArgs):
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "optimizer": "adafactor",
                "dataset": "t0",
                "model": "t5-small",
            }
        )
    )
    config = SimpleArgs.from_json(str(config_file))
    assert not config.was_overridden("train_dir")
    assert not config.was_overridden("optimizer")
    assert config.was_default("dataset")
    assert config.optimizer == "adafactor"
    assert config.dataset == "t0"
    assert config.model == "t5-small"


def test_config_was_default_from_kwargs(SimpleArgs):
    config = SimpleArgs(**{"dataset": "t1"})
    assert not config.was_default("dataset")
    assert config.was_default("model")


def test_config_was_default_from_file(tmp_path, SimpleArgs):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"dataset": "t1"}))
    config = SimpleArgs.from_json(config_file)
    assert not config.was_default("dataset")
    assert config.was_default("model")


def test_auto_modifier_config():
    from mttl.config import ModifierArgs
    from mttl.models.modifiers.base import ModifierConfig

    config = ModifierArgs()
    config.model_modifier = "lora"
    config.lora_rank = 12
    config.lora_dropout = 0.52
    config.modify_modules = ".*mlpU.*"

    lora_config = ModifierConfig.from_training_config(config)

    from mttl.models.modifiers.lora import LoRAConfig

    assert type(lora_config) == LoRAConfig
    assert lora_config.lora_rank == 12
    assert lora_config.lora_dropout == 0.52
    assert lora_config.modify_modules == ".*mlpU.*"


def test_dump_load_lora_config():
    from mttl.models.modifiers.base import ModifierConfig

    data = {
        "__model_modifier__": "lora",
        "lora_rank": 12,
        "lora_dropout": 0.52,
    }
    lora_config = ModifierConfig.fromdict(data)

    from mttl.models.modifiers.lora import LoRAConfig

    assert type(lora_config) == LoRAConfig
    assert lora_config.lora_rank == 12
    assert lora_config.lora_dropout == 0.52


def test_dump_load_selector_config():
    from mttl.models.containers.selectors.base import SelectorConfig
    from mttl.models.containers.selectors.moe_selector import MOERKHSSelectorConfig

    dump = MOERKHSSelectorConfig(emb_dim=12345).asdict()
    test = SelectorConfig.fromdict(dump)
    assert test.emb_dim == 12345
    assert type(test) == MOERKHSSelectorConfig
