import ast
import json
import os
from argparse import ArgumentParser
from dataclasses import MISSING, asdict, dataclass, field, fields, make_dataclass
from typing import Any

from mttl.datamodule.base import DatasetConfig, DefaultDataModule
from mttl.logging import logger, setup_logging
from mttl.models.containers.selectors.base import Selector, SelectorConfig
from mttl.models.modifiers.base import Modifier, ModifierConfig
from mttl.registrable import Registrable
from mttl.serializable import Serializable

# sentinel object to detect if an argument field needs
# to inherit the default value from a dataclass
INHERIT_DEFAULT = object()


@dataclass
class MttlArgs(Serializable):
    def to_config(self):
        return self


class RegistrableArgs(type):
    def __new__(cls, name, bases, attrs, registrable_cls: Registrable, arg_name: str):
        new_fields = {}

        for c in registrable_cls.registered_configs():
            for f in fields(c):
                if f.name == arg_name:
                    raise ValueError(
                        f"Field name '{f.name}' is reserved for the argument parser."
                    )
                new_fields[f.name] = (f.type, field(default=INHERIT_DEFAULT))

        new_fields[arg_name] = (str, field(default=None))

        def to_config(self):
            arg_value = getattr(self, arg_name)

            # user did not provide a value for this argument
            if arg_value is None:
                return None

            # get the config class from the registry corresponding to the selected type
            config_cls: dataclass = registrable_cls.get_config_class_by_name(arg_value)

            # get args from the current dataclass that are not INHERIT_DEFAULT
            kwargs = {}
            for k in fields(config_cls):
                value = getattr(self, k.name)
                if value is not INHERIT_DEFAULT:
                    # it has been set by the user!
                    kwargs[k.name] = value
            return config_cls(**kwargs)

        def post_init(self):
            allowed_values = registrable_cls.registered_names()
            if getattr(self, arg_name) not in allowed_values:
                raise ValueError(
                    f"Invalid value for {arg_name}. Must be one of {allowed_values}"
                )

        to_tuples = [(k,) + v for k, v in new_fields.items()]
        new_cls = make_dataclass(name, to_tuples, bases=(MttlArgs,))

        # set functions to be had in the new baby dataclass
        for k, v in {
            **attrs,
            "to_config": to_config,
            "__post_init__": post_init,
        }.items():
            setattr(new_cls, k, v)
        return new_cls


@dataclass
class TrainingArgs(MttlArgs):
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

    # evaluation stuff
    pipeline_eval_tasks: str = None


@dataclass
class ModelArgs(MttlArgs):
    model: str = None
    model_family: str = "gpt"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    attn_implementation: str = None


@dataclass
class ModifierArgs(
    metaclass=RegistrableArgs,
    registrable_cls=Modifier,
    arg_name="model_modifier",
):
    pass


@dataclass
class DatasetArgs(
    metaclass=RegistrableArgs,
    registrable_cls=DefaultDataModule,
    arg_name="dataset_type",
):
    pass


@dataclass
class SelectorArgs(
    metaclass=RegistrableArgs,
    registrable_cls=Selector,
    arg_name="router_selector",
):
    pass


class MttlArgumentParser:
    def __init__(
        self,
        dataclasses_list=(
            TrainingArgs,
            ModelArgs,
            DatasetArgs,
            SelectorArgs,
            ModifierArgs,
        ),
    ):
        self.dataclasses_list = dataclasses_list
        for dc in self.dataclasses_list:
            if not issubclass(dc, MttlArgs):
                raise ValueError(
                    f"{dc} must be a subclass of MttlArgs to be used in the argument parser"
                )

        self.parser = ArgumentParser()
        self.parser.add_argument(
            "-c",
            "--config",
            type=str,
            help="Path to the configuration file",
            default=None,
        )

        unique_fields = {}
        for dc in self.dataclasses_list:
            for f in fields(dc):
                # inherits default values from other dataclasses
                if f.default != INHERIT_DEFAULT:
                    if (
                        f.name in unique_fields
                        and f.default != unique_fields[f.name][1]
                    ):
                        logger.warning(
                            f"The field {dc}.{f.name} has a different default in a different dataclass!"
                        )
                else:
                    if f.default == INHERIT_DEFAULT and f.name in unique_fields:
                        continue

                unique_fields[f.name] = (f.type, f.default)

        for k, v in unique_fields.items():
            self.parser.add_argument(f"--{k}", type=v[0], default=v[1])

    def update_kwargs(self, args, overrides, eval=True, raise_error=True, silent=False):
        overwrites_log = []
        for k, v in overrides.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    v = v
            else:
                v = v

            if not hasattr(args, k) and raise_error:
                raise ValueError(f"{k} is not in the config")

            if eval and not silent:
                overwrites_log.append(f"Overwriting {k} to {v}")

            if type(v) == str and "$" in v:
                from string import Template

                # this raises an error if the env. var does not exist
                v = Template(v).substitute(os.environ)

            setattr(args, k, v)
        return overwrites_log

    def parse_args(self, raise_error=True, silent=False, **kwargs):
        args = self.parser.parse_args()

        overwrite_logs = []

        if args.config:
            if not os.path.exists(args.config):
                filename = os.path.join(
                    os.getenv("CONFIG_PATH", default="configs"), filename
                )

                if not os.path.exists(filename) and ".json" not in filename:
                    filename = filename + ".json"
            else:
                filename = args.config

            overwrite_logs += self.update_kwargs(
                args,
                json.load(open(filename)),
                eval=False,
                raise_error=raise_error,
                silent=silent,
            )

        # setup logging to the output dir
        try:
            setup_logging(args.output_dir)
        except Exception as e:
            if raise_error:
                raise ValueError("Error setting up logging") from e
            elif not silent:
                logger.warning(f"Error setting up logging to {self.output_dir}")

        # log the overwrites
        for log in overwrite_logs:
            logger.warning(log)

        self.post_init(args, silent=silent)
        return args

    def parse_args_into_dataclasses(self):
        args = self.parse_args()

        outputs = []
        for dc in self.dataclasses_list:
            dc_args = {}

            for f in fields(dc):
                dc_args[f.name] = getattr(args, f.name)

            dc = dc(**dc_args).to_config()
            outputs.append(dc)
        return outputs

    def post_init(self, args, silent=False):
        if args.attn_implementation == "eager" and args.pack_sequences:
            logger.warning(
                "Eager attention is not compatible with packed sequences"
                + ", tokens across examples will not be masked"
            )
        elif args.attn_implementation == "flash_attention_2" and args.pack_sequences:
            logger.warning(
                "The wrapper we provide for flash attention 2 may not behave as expected for"
                + " some models. Please make sure you test the model with packed sequences"
            )
