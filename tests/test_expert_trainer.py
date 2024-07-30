import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from conftest import tmp_exp_config
from pytorch_lightning import seed_everything
from transformers import AutoModelForCausalLM

from mttl.callbacks import LiveCheckpointCallback
from mttl.config import Config
from mttl.datamodule.alpaca_data_module import AlpacaDataModule
from mttl.datamodule.base import DatasetConfig
from mttl.models.containers import get_modules_to_modify_trie
from mttl.models.containers.selectors import (
    ArrowSelector,
    PolySelector,
    PolySelectorConfig,
    TaskNameSelector,
    TaskNameSelectorConfig,
)
from mttl.models.containers.selectors.moe_selector import MOERKHSSelectorConfig
from mttl.models.expert_model import Expert, ExpertModel, LoRAMoEModel, MultiExpertModel
from mttl.models.expert_trainer import (
    ExpertModelLightningWrapper,
    LoRAMoELightningWrapper,
    MultiExpertModelLightningWrapper,
)
from mttl.models.modifiers.lora import LoRA, LoRAConfig, SkilledLoRAConfig


@pytest.fixture
def alpaca():
    yield AlpacaDataModule(
        DatasetConfig(
            dataset="alpaca",
            model="EleutherAI/gpt-neo-125m",
            model_family="gpt",
            validation_portion=0.0001,
            train_batch_size=2,
        ),
        for_generation=False,
        val_mixin=False,
    )


def test_expert_trainer(tmp_path, alpaca, tmp_exp_config):
    seed_everything(0)

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=tmp_path,
        monitor="val/loss",
        save_last=True,
        mode="min",
        save_each_epoch=True,
    )
    model = ExpertModel(
        "EleutherAI/gpt-neo-125m",
        modifier_config=LoRAConfig(modify_layers=".*out_proj.*"),
        device_map="cpu",
    )

    config: Config = tmp_exp_config

    wrapper = ExpertModelLightningWrapper(model, config)
    trainer = pl.Trainer(
        accelerator="cpu",
        default_root_dir=tmp_path,
        max_steps=1,
        val_check_interval=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(wrapper, datamodule=alpaca)

    # reload the model without the wrapper, directly from checkpoint!
    loaded_model = ExpertModel.from_pretrained(tmp_path / "last.ckpt")
    for n, p in loaded_model.named_parameters():
        assert torch.allclose(p, wrapper.model.state_dict()[n], atol=1e-4)

    # reload the trainer from the same checkpoint! (i.e. we might want to continue training...)
    loaded_wrapper = ExpertModelLightningWrapper.load_from_checkpoint(
        tmp_path / "last.ckpt"
    )
    for n, p in loaded_wrapper.named_parameters():
        assert torch.allclose(p, wrapper.state_dict()[n], atol=1e-4)


def test_multi_expert_trainer(tmp_path, alpaca, tmp_exp_config):
    seed_everything(0)

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=tmp_path,
        monitor="val/loss",
        save_last=True,
        mode="min",
        save_each_epoch=True,
    )
    config: Config = tmp_exp_config
    tmp_exp_config.trainable_param_names = ".*lora.*|.*rkhs.*"

    # multi expert model now
    model = MultiExpertModel(
        "EleutherAI/gpt-neo-125m",
        selector_config=MOERKHSSelectorConfig(
            router_granularity="finegrained", rkhs_dim=5, emb_dim=5, top_k=1
        ),
        device_map="cpu",
    )
    model.add_empty_expert(
        "a1",
        LoRAConfig(modify_layers=".*out_proj.*"),
    )
    model.add_empty_expert(
        "a2",
        LoRAConfig(modify_layers=".*out_proj.*"),
    )

    wrapper = MultiExpertModelLightningWrapper(model, config)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=1,
        default_root_dir=tmp_path,
        val_check_interval=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(wrapper, datamodule=alpaca)

    # reload the model without the wrapper, directly from checkpoint!
    loaded_model = MultiExpertModel.from_pretrained(tmp_path / "last.ckpt")
    for n, p in loaded_model.named_parameters():
        assert torch.allclose(p, wrapper.model.state_dict()[n], atol=1e-4)

    # reload the trainer from the same checkpoint! (i.e. we might want to continue training...)
    loaded_wrapper = MultiExpertModelLightningWrapper.load_from_checkpoint(
        tmp_path / "last.ckpt"
    )
    for n, p in loaded_wrapper.named_parameters():
        assert torch.allclose(p, wrapper.state_dict()[n], atol=1e-4)


def test_lora_moe_trainer(tmp_path, alpaca, tmp_exp_config):
    seed_everything(0)

    checkpoint_callback = LiveCheckpointCallback(
        dirpath=tmp_path,
        monitor="val/loss",
        save_last=True,
        mode="min",
        save_each_epoch=True,
    )
    model = ExpertModel(
        "EleutherAI/gpt-neo-125m",
        modifier_config=LoRAConfig(modify_layers=".*out_proj.*"),
        device_map="cpu",
    )

    config: Config = tmp_exp_config
    tmp_exp_config.trainable_param_names = ".*lora.*|.*rkhs.*"

    # multi expert model now
    model = LoRAMoEModel(
        "EleutherAI/gpt-neo-125m",
        modifier_config=SkilledLoRAConfig(n_splits=1, modify_layers=".*out_proj.*"),
        selector_config=MOERKHSSelectorConfig(
            router_granularity="finegrained",
            num_experts=2,
            rkhs_dim=5,
            emb_dim=5,
            top_k=1,
        ),
        device_map="cpu",
    )

    wrapper = LoRAMoELightningWrapper(model, config)
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=1,
        default_root_dir=tmp_path,
        val_check_interval=1,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(wrapper, datamodule=alpaca)

    # reload the model without the wrapper, directly from checkpoint!
    loaded_model = LoRAMoEModel.from_pretrained(tmp_path / "last.ckpt")
    for n, p in loaded_model.named_parameters():
        assert torch.allclose(p, wrapper.model.state_dict()[n], atol=1e-4)

    # reload the trainer from the same checkpoint! (i.e. we might want to continue training...)
    loaded_wrapper = MultiExpertModelLightningWrapper.load_from_checkpoint(
        tmp_path / "last.ckpt"
    )
    for n, p in loaded_wrapper.named_parameters():
        assert torch.allclose(p, wrapper.state_dict()[n], atol=1e-4)
