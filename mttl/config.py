import argparse
import ast
import importlib
import json
import os
from dataclasses import MISSING, Field, dataclass, field, fields, make_dataclass
from string import Template
from typing import Any, Dict, List, Type

import torch

from mttl.logging import logger, setup_logging, warn_once
from mttl.registrable import Registrable


class MultiDefaultValue:
    """
    Manages multiple default values for fields that may have the same name but different defaults
    across merged dataclasses.
    """

    def __init__(self, cls, name, field_type, default):
        self.name = name
        self.type = field_type
        self.defaults = {cls.__name__: default}

    def update(self, cls, default, field_type):
        if field_type != self.type:
            raise TypeError(
                f"Field '{self.name}' has conflicting types: {field_type} and {self.type}."
            )

        # Add a new default only if it's different from the last one
        last_default = list(self.defaults.values())[-1]
        if default != last_default:
            self.defaults[cls.__name__] = default

    @property
    def default(self):
        return next(iter(self.defaults.values())) if len(self.defaults) == 1 else self

    def __repr__(self):
        if len(self.defaults) == 1:
            return repr(self.default)
        return f"MultiDefaultValue({self.defaults})"


def dataclasses_union(*args: Type[dataclass]) -> List:
    """
    Create a new dataclass that is the union of the fields of the input dataclasses.
    """
    new_fields = {}
    for c in args:
        for f in fields(c):
            if f.name not in new_fields:
                value = (
                    f.type,
                    field(default=MultiDefaultValue(c, f.name, f.type, f.default)),
                )
                new_fields[f.name] = value
            else:
                value = new_fields[f.name][1]
                value.default.update(c, f.default, f.type)

    # now we need to set the default values for all the fields in which there is only one default value
    for k in new_fields:
        value = new_fields[k][1]
        value.default = value.default.default

    return [(k,) + v for k, v in new_fields.items()]


def create_config_class_from_args(config_class: dataclass, args: "Args"):
    """
    Load a dataclass from the arguments.
    """
    kwargs = {}

    for f in fields(config_class):
        if hasattr(args, f.name):
            value = getattr(args, f.name)

            # if the field is not a multi-default, we override it (it was either default or set by the user)
            if not isinstance(value, MultiDefaultValue):
                kwargs[f.name] = value

    return config_class(**kwargs)


@dataclass
class Args:
    @property
    def updated_kwargs(self):
        return {
            k.name: getattr(self, k.name)
            for k in fields(self)
            if getattr(self, k.name) != k.default
        }

    @classmethod
    def fromdict(cls, data, strict=False) -> "Args":
        """
        Reload a dataclass from a dictionary. We store the class name in the dict to be able to reload it.
        If the cls is Args, then we try to access the `_class` attribute to get the class name.
        """
        cls_name = data.pop("args_class", None)
        if cls == Args:
            if not cls_name:
                raise ValueError("No class name found in the data.")
            cls = globals()[cls_name]

        if not strict:
            data_ = {}
            for f in fields(cls):
                if f.name in data:
                    data_[f.name] = data[f.name]
        else:
            data_ = data

        return cls(**data_)

    def asdict(self) -> Dict:
        from mttl.models.utils import convert_hps_to_dict

        data = convert_hps_to_dict(self.__dict__)
        return {"args_class": self.__class__.__name__, **data}

    def was_overridden(self, key):
        return key in self.updated_kwargs

    def was_default(self, key):
        return key not in self.updated_kwargs

    @classmethod
    def process_kwargs(cls, kwargs, eval=True, raise_error=True, silent=False):
        overwrites_log = []

        for k, v in kwargs.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    v = v
            else:
                v = v

            if not hasattr(cls, k) and raise_error:
                raise ValueError(f"{k} is not in the config")

            if eval and not silent:
                overwrites_log.append(f"Overwriting {k} to {v}")

            if type(v) == str and "$" in v:
                # this raises an error if the env. var does not exist
                v = Template(v).substitute(os.environ)

            kwargs[k] = v
        return overwrites_log

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        import copy

        to_save = copy.deepcopy(self.__dict__)
        return json.dumps(to_save, indent=4, sort_keys=False)

    @classmethod
    def from_json(cls, config_file, raise_error=True):
        """
        Loads the config from a file
        """
        with open(config_file, "r") as fin:
            kwargs = json.load(fin)

        overwrite_logs = cls.process_kwargs(
            kwargs,
            eval=False,
            raise_error=raise_error,
        )
        # log the overwrites
        for log in overwrite_logs:
            logger.warning(log)

        return cls(**kwargs)

    def save_config(self, output_dir):
        """
        Saves the config
        """
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.json"), "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")

    @classmethod
    def parse(cls, raise_error=True, return_parser=False):
        import itertools

        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config_files", required=False)
        parser.add_argument("-k", "--kwargs", nargs="*", action="append")
        args = parser.parse_args()

        cmd_kwargs = {}
        if args.kwargs:
            kwargs_opts = list(itertools.chain(*args.kwargs))
            for value in kwargs_opts:
                key, _, value = value.partition("=")

                # allows multiple values for a given option when specified in the command line!
                if key in cmd_kwargs:
                    if type(cmd_kwargs[key]) != list:
                        cmd_kwargs[key] = [cmd_kwargs[key]]
                    cmd_kwargs[key].append(value)
                else:
                    cmd_kwargs[key] = value

        overwrite_logs = []
        extra_kwargs = {}

        if args.config_files:
            for filename in args.config_files.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(
                        os.getenv("CONFIG_PATH", default="configs"), filename
                    )

                if not os.path.exists(filename) and ".json" not in filename:
                    filename = filename + ".json"

                file_kwargs = json.load(open(filename))
                overwrite_logs += cls.process_kwargs(
                    file_kwargs,
                    eval=False,
                    raise_error=raise_error,
                )
                extra_kwargs.update(file_kwargs)

        if cmd_kwargs:
            overwrite_logs += cls.process_kwargs(
                cmd_kwargs,
                raise_error=raise_error,
            )
            extra_kwargs.update(cmd_kwargs)

        # log the overwrites
        for log in overwrite_logs:
            logger.warning(log)

        config = cls(**extra_kwargs)

        if return_parser:
            return config, args
        return config


class MetaRegistrable(type):
    """
    Meta class that creates a new dataclass containing all the configs
    in this registrable.
    """

    def __new__(cls, name, bases, attrs, registrable: Registrable = None):
        module_name, class_name = registrable.rsplit(".", 1)
        module = importlib.import_module(module_name)

        registrable_class = getattr(module, class_name)

        # make the union of all the fields across the registered configs
        to_tuples = dataclasses_union(*registrable_class.registered_configs())

        # create new dataclass with the union of all the fields
        new_cls = make_dataclass(name, to_tuples, bases=(Args,), init=False)
        new_cls.registrable_class = registrable_class

        # set functions to be had in the new baby dataclass
        for k, v in {
            **attrs,
        }.items():
            setattr(new_cls, k, v)
        return new_cls


@dataclass
class DataArgs(
    metaclass=MetaRegistrable, registrable="mttl.datamodule.base.DataModule"
):
    pass


@dataclass
class SelectorArgs(
    metaclass=MetaRegistrable,
    registrable="mttl.models.containers.selectors.base.Selector",
):
    pass


@dataclass
class ModifierArgs(
    metaclass=MetaRegistrable, registrable="mttl.models.modifiers.base.Modifier"
):
    pass


@dataclass
class TransformArgs(
    metaclass=MetaRegistrable,
    registrable="mttl.models.library.library_transforms.LibraryTransform",
):
    pass


@dataclass
class TrainingArgs(DataArgs):
    cache_dir: str = os.getenv("CACHE_DIR", "./cache")
    data_dir: str = os.getenv("TRAIN_DIR", "/tmp/")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")

    finetune_task_name: str = None
    exp_name: str = None
    expert_name: str = None

    # Training config
    micro_batch_size: str = None
    compute_strategy: str = None
    scheduler: str = "linear_decay_with_warmup"
    checkpoint: str = None  # load from checkpoint
    checkpoint_step: str = None  # load from checkpoint in format of global_stepX.pt
    backbone_checkpoint: str = None  # load the backbone from here
    learning_rate: float = 1e-3
    warmup_proportion: float = 0.06
    trainable_param_names: str = ".*"
    non_trainable_param_names: str = None
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.1
    gradient_accumulation_steps: int = 1
    optimizer: str = "adamw"
    adafactor_scale_parameter: bool = True
    adafactor_warmup_init: bool = False
    adafactor_relative_step: bool = False
    num_train_epochs: int = -1
    warmup_steps: int = -1
    total_steps: int = -1
    num_tasks_per_batch: int = None
    save_every: int = None
    save_each_epoch: bool = False
    eval_every: int = None
    eval_every_n_epoch: int = 1
    seed: int = 42
    debug: bool = False

    eval_before_training: bool = True
    precision: str = "32"
    monitor_grad_alignment_on: str = None

    model: str = None
    model_family: str = None
    attn_implementation: str = None
    device_map: str = "cpu"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    do_train: bool = True

    # logging
    wandb_project: str = None
    tensorboard: bool = False
    remote_token: str = None
    library_id: str = None
    destination_library_id: str = None
    logging_prefix: str = ""

    router_weight_decay: float = None  # router weight decay
    router_learning_rate: float = None

    module_logits_relaxed_bernoulli: bool = True
    module_logits_straight_through: bool = False
    module_logits_learning_rate: float = 0.1
    adapters_learning_rate: float = None
    adapters_weight_decay: float = None
    module_logits_dropout: float = 0.0
    module_logits_l2_norm: float = False

    # some eval flags during training
    eval_mmlu_few_shot: bool = True  # use few-shot for mmlu, default
    eval_mmlu_flag: bool = False  # eval mmlu performance during training
    eval_rouge_flag: bool = False  # eval rouge during training
    eval_before_training: bool = True
    pipeline_eval_tasks: str = None
    save_if_loaded_from_ckpt: bool = True
    dataset_type: str = None

    @property
    def dataset_config(self):
        if self.dataset_type is not None:
            return create_config_class_from_args(
                DataArgs.registrable_class.get_config_class_by_name(self.dataset_type),
                self,
            )

    def __post_init__(self):
        if self.attn_implementation == "eager" and self.pack_sequences:
            logger.warning(
                "Eager attention is not compatible with packed sequences"
                + ", tokens across examples will not be masked"
            )
        elif self.attn_implementation == "flash_attention_2" and self.pack_sequences:
            logger.warning(
                "The wrapper we provide for flash attention 2 may not behave as expected for"
                + " some models. Please make sure you test the model with packed sequences"
            )

        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size

        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            warn_once(
                "You have multiple GPUs, but your device count is not being taken "
                + "into account when computing `gradient_accumulation_steps`."
            )


@dataclass
class ExpertConfig(TrainingArgs, ModifierArgs):
    model_modifier: str = None

    @property
    def modifier_config(self):
        from mttl.models.modifiers.base import ModifierConfig

        return ModifierConfig.from_training_config(self)


@dataclass
class MultiExpertConfig(ExpertConfig, SelectorArgs):
    router_selector: str = None

    @property
    def selector_config(self):
        from mttl.models.containers.selectors.base import MultiSelectorConfig

        return MultiSelectorConfig.from_training_config(self)


@dataclass
class FinetuneConfig(MultiExpertConfig):
    finetune_type: str = None  # ["F", "A", "Z", "MuZ", "Poly", "PolyRand"]
    finetune_skip_es: str = False  # skip early stopping while fine-tuning
    finetune_use_last_checkpoint: bool = (
        False  # use always the best valid_perf checkpoint if available
    )
    expert_selection: str = None
    use_vllm: str = False
    # for finetuning a library
    hf_repo_query: str = None  # for retrieval, we take query expert from this library
    sk: int = 5  # number of experts to retrieve from a library
    finetune_regime: str = None  # polylib_full, lib_mu, polylib_selector
    tasksets_path: str = None

    def __post_init__(self):
        if self.finetune_task_name is not None and isinstance(
            self.finetune_task_name, str
        ):
            # resolve task keys
            task_names = []
            tasks = self.finetune_task_name.split(",")
            if self.tasksets_path is not None and os.path.exists(self.tasksets_path):
                # load task names from json file
                task_sets = json.load(open(self.tasksets_path))
                for task_name in tasks:
                    # try to fetch task_names from the file
                    if task_name in task_sets:
                        task_names.extend(task_sets[task_name])
                    else:
                        task_names.append(task_name)
            else:
                task_names = tasks

            self.finetune_task_name = ",".join(task_names)


@dataclass
class EvaluationConfig(MultiExpertConfig, TransformArgs):
    expert_selection: str = None
    load_module: str = None
    eval_metric: str = "loss"
    merge_or_route: str = None  # "uniform", "ties", "clown"
    tasksets_path: str = None
    remove_experts: str = None
    create_transfer_matrix: bool = False
    es_metric: str = "loss"
    n_ng_iterations: int = 30  # number of iterations for LoraHub
    recompute_prototypes: bool = False


@dataclass
class MoEExpertConfig(MultiExpertConfig):
    moe_ent_reg: float = 0.0
    moe_ent_free_bits: float = 0.0
    moe_num_experts: int = 8
