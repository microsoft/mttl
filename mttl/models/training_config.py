import os
from dataclasses import MISSING, asdict, dataclass, field, fields, make_dataclass
from typing import Any

from mttl.datamodule.base import DatasetConfig, DefaultDataModule
from mttl.models.containers.selectors.base import Selector, SelectorConfig
from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.serializable import Serializable


@dataclass
class TrainingConfig(Serializable):
    data_dir: str = os.getenv("TRAIN_DIR", "/tmp/")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")

    exp_name: str = None
    wandb_project: str = None
    freeze_embeds: bool = False

    # training config
    compute_strategy: str = None
    precision: str = "32"
    checkpoint: str = None  # load from checkpoint

    # logging stuff
    num_train_epochs: int = -1
    warmup_steps: int = -1
    total_steps: int = -1
    save_every: int = None
    eval_every: int = None
    eval_every_n_epoch: int = 1
    debug: bool = False
    seed: int = 42
    eval_before_training: bool = True

    # optimizer flags
    optimizer: str = "adamw"
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = False
    adafactor_relative_step: bool = False
    scheduler: str = "linear_decay_with_warmup"
    learning_rate: str = 1e-3
    warmup_proportion: str = 0.06
    trainable_param_names: str = ".*"
    non_trainable_param_names: str = None
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.1
    micro_batch_size: int = 1

    finetune_type: str = None  # ["F", "A", "Z", "MuZ", "Poly", "PolyRand"]
    finetune_skip_es: bool = False  # skip early stopping while fine-tuning
    finetune_use_last_checkpoint: bool = (
        False  # use always the best valid_perf checkpoint if available
    )
    monitor_grad_alignment_on: str = None

    # routing flags
    router_weight_decay: float = 0.0  # router weight decay
    router_learning_rate: float = 0.0

    # adapter flags
    adapters_learning_rate: float = None
    adapters_weight_decay: float = None
    module_logits_dropout: float = 0.0
    module_logits_l2_norm: bool = False

    pipeline_eval_tasks: str = None


@dataclass
class TFewTrainingConfig(TrainingConfig):
    mc_loss: float = 0.0  # T-Few
    length_norm: float = 0.0  # T-Few
    unlikely_loss: float = 0.0  # T-Few


@dataclass
class FinetuningConfig(TrainingConfig):
    mc_loss: float = 0.0  # T-Few
    length_norm: float = 0.0  # T-Few
    unlikely_loss: float = 0.0  # T-Few


@dataclass
class ModelConfig(Serializable):
    model: str = None
    model_family: str = "gpt"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    attn_implementation: str = None


class MetaConfig(type):
    """Creates a new dataclass with all fields from all registered configs."""

    @property
    def selector_config(self):
        return SelectorConfig.from_dict(self.to_dict())

    @property
    def modifier_config(self):
        return ModifierConfig.from_dict(self.to_dict())

    @property
    def dataset_config(self):
        return DatasetConfig.from_dict(self.to_dict())

    def __new__(cls, name, bases, attrs, training_config, model_config):
        from dataclasses import MISSING

        reserved = ["model_modifier", "dataset", "router_selector"]

        inherit_default_configs = [training_config, model_config]
        none_default_configs = (
            Modifier.registered_configs()
            + Selector.registered_configs()
            + DefaultDataModule.registered_configs()
        )

        new_fields = {}
        for dataclass_type in none_default_configs:
            for f in fields(dataclass_type):
                if f.name in reserved:
                    raise ValueError(
                        f"Field name '{f.name}' is reserved for the config class."
                    )
                new_fields[f.name] = (f.type, field(default=None))

        for dataclass_type in inherit_default_configs:
            for f in fields(dataclass_type):
                if f.name in reserved:
                    raise ValueError(
                        f"Field name '{f.name}' is reserved for the config class."
                    )
                new_fields[f.name] = (f.type, field(default=f.default))

        # these are reserved fields
        new_fields["dataset"] = (str, field(default=None))
        new_fields["model_modifier"] = (str, field(default=None))
        new_fields["router_selector"] = (str, field(default=None))

        to_tuples = [(k,) + v for k, v in new_fields.items()]

        # bring over the annotations from the class
        new_cls = make_dataclass(name, to_tuples, bases=(Serializable,))

        for field_name, field_type in attrs.items():
            setattr(new_cls, field_name, field_type)

        setattr(new_cls, "selector_config", cls.selector_config)
        setattr(new_cls, "modifier_config", cls.modifier_config)
        setattr(new_cls, "dataset_config", cls.model_config)

        def get_training_config(self):
            return training_config.from_dict(self.to_dict())

        def get_model_config(self):
            return model_config.from_dict(self.to_dict())

        setattr(new_cls, "training_config", property(get_training_config))
        setattr(new_cls, "model_config", property(get_model_config))
        return new_cls


class TFewGlobalConfig(
    metaclass=MetaConfig,
    training_config=TrainingConfig,
    model_config=ModelConfig,
):
    pass
