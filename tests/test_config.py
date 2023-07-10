import json

from mttl.projects.mhr.mhr_config import MHRConfig as Config

def test_config_dict_like(tmp_path):
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
    config = Config(kwargs=config_dict)
    reconstructed_config = json.loads(json.dumps(config.__dict__))
    assert optimizer in reconstructed_config["optimizer"]
    assert dataset in reconstructed_config["dataset"]
    assert model in reconstructed_config["model"]
    assert train_dir in reconstructed_config["train_dir"]
    assert total_steps == reconstructed_config["total_steps"]
    assert learning_rate == reconstructed_config["learning_rate"]


def test_config_was_override_from_kwargs():
    config = Config(kwargs={
        "optimizer": "adafactor",
        "dataset": "t0",
        "model": "t5-small",
    })
    assert not config.was_overridden("train_dir")
    assert config.was_overridden("optimizer")
    assert config.was_overridden("dataset")
    assert config.was_overridden("model")
    assert config.optimizer == "adafactor"
    assert config.dataset == "t0"
    assert config.model == "t5-small"


def test_config_was_override_from_file(tmp_path):
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
    config = Config(filenames=str(config_file))
    assert not config.was_overridden("train_dir")
    assert config.was_overridden("optimizer")
    assert config.was_overridden("dataset")
    assert config.was_overridden("model")
    assert config.optimizer == "adafactor"
    assert config.dataset == "t0"
    assert config.model == "t5-small"


def test_config_was_default_from_kwargs():
    config = Config(kwargs={"dataset": "t0"})
    assert not config.was_default("dataset")
    assert config.was_default("model")


def test_config_was_default_from_file(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"dataset": "t0"}))
    config = Config(filenames=str(config_file))
    assert not config.was_default("dataset")
    assert config.was_default("model")
