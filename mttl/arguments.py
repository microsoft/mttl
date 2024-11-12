import argparse
import ast
import importlib
import json
import os
from dataclasses import MISSING, dataclass, field, fields, make_dataclass
from string import Template
from typing import Dict, List, Type, TypeVar

import torch

# import registrables for args management
from mttl.logging import logger, warn_once
from mttl.registrable import Registrable
from mttl.serializable import AutoSerializable, Serializable
from mttl.utils import deprecated

# Create a generic type variable that can be any type
T = TypeVar("T")


class MultiDefaultValue:
    """
    When we merge dataclasses fields to compile the available command-line args, different dataclasses might have
    different defaults for the same field. This class is used to store all the defaults and resolve them when needed.
    """

    def __init__(self, field_type: Type[T]):
        self.type = field_type
        self.defaults: Dict[str, T] = {}

    def add_default(self, klass, value, field_type):
        if field_type != self.type:
            raise TypeError(
                f"Field has conflicting types: {field_type} and {self.type}."
            )

        if value not in set(self.defaults.values()):
            self.defaults[klass.__name__] = value

    def resolve(self):
        if len(self.defaults) == 1:
            return list(self.defaults.values())[0]
        return self

    def __repr__(self):
        return f"MultiDefaultValue({self.defaults})"


def dataclasses_union(*dataclasses: Type[dataclass]) -> List:
    """
    Create a new dataclass that is the union of the fields of the input dataclasses.
    """
    new_fields = {}
    for klass in dataclasses:
        for field_ in fields(klass):
            name = field_.name

            if field_.default_factory is not MISSING:
                raise ValueError(
                    "Attributes with default factories are not supported yet!"
                )

            if field_.name not in new_fields:
                # If the field does not exist, we create it
                multi_default = MultiDefaultValue(field_.type)
            else:
                multi_default = new_fields[name][1].default

            multi_default.add_default(klass, field_.default, field_.type)
            new_fields[name] = (multi_default.type, field(default=multi_default))

    # try to resolve the MultiDefaultValue objects
    for name, (field_type, field_info) in new_fields.items():
        multi_default = field_info.default
        if isinstance(multi_default, MultiDefaultValue):
            new_fields[name] = (field_type, field(default=multi_default.resolve()))

    return [(name,) + field_info for name, field_info in new_fields.items()]


def create_config_class_from_args(config_class, args):
    """
    Load a dataclass from the arguments. We don't include field names that were not set,
    i.e. that were MultiDefaultValue.
    """
    kwargs = {
        f.name: getattr(args, f.name)
        for f in fields(config_class)
        if hasattr(args, f.name)
        and not isinstance(getattr(args, f.name), MultiDefaultValue)
    }

    return config_class(**kwargs)


@dataclass
class Args(Serializable):
    @property
    def updated_kwargs(self):
        return {
            k.name: getattr(self, k.name)
            for k in fields(self)
            if getattr(self, k.name) != k.default
        }

    def asdict(self) -> Dict:
        """Slightly overloaded asdict to skip MultiDefaultValue fields."""
        skip_fields = [
            f.name
            for f in fields(self)
            if isinstance(getattr(self, f.name), MultiDefaultValue)
        ]
        return super().asdict(skip_fields=skip_fields)

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

        to_save = self.asdict()
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


class AutoArgs(AutoSerializable):
    @classmethod
    @deprecated(
        message="The config appears to be a legacy config and will be discontinued in the next release."
    )
    def fromdict_legacy(cls, data):
        """Assume the data is ExpertConfig to not break previous loading."""
        dataclass_cls = ExpertConfig
        return dataclass_cls.fromdict(data)

    @classmethod
    def fromdict(cls, data: Dict):
        try:
            return AutoSerializable.fromdict(data)
        except ValueError:
            return cls.fromdict_legacy(data)


class FromRegistrable(type):
    """
    Meta class that creates a new dataclass containing fields all the config dataclasses
    in this registrable.
    """

    def __new__(cls, name, bases, attrs, registrable: str = None):
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
    metaclass=FromRegistrable, registrable="mttl.datamodule.base.DataModule"
):
    pass


@dataclass
class SelectorArgs(
    metaclass=FromRegistrable,
    registrable="mttl.models.containers.selectors.base.Selector",
):
    pass


@dataclass
class ModifierArgs(
    metaclass=FromRegistrable, registrable="mttl.models.modifiers.base.Modifier"
):
    pass


@dataclass
class TransformArgs(
    metaclass=FromRegistrable,
    registrable="mttl.models.library.library_transforms.LibraryTransform",
):
    pass


@dataclass
class TrainingArgs(DataArgs):
    # model arguments
    model: str = None
    model_family: str = None
    attn_implementation: str = None
    device_map: str = "cpu"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    do_train: bool = True

    # output directories
    cache_dir: str = os.getenv("CACHE_DIR", "./cache")
    data_dir: str = os.getenv("TRAIN_DIR", "/tmp/")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")

    # meta-tasks or group of tasks
    finetune_task_path: str = None
    # name of tasks, or name of group of tasks if finetune_task_path is set
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
    trainable_param_names: str = (
        ".*"  # trainable param names are those set by the experts
    )
    non_trainable_param_names: str = None
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.1
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

    precision: str = "32"
    monitor_grad_alignment_on: str = None

    # logging
    wandb_project: str = None
    wandb_run_name: str = None
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
    create_transfer_matrix: bool = False
    pipeline_eval_tasks: str = None
    save_if_loaded_from_ckpt: bool = True
    dataset_type: str = None

    profile: bool = False  # if 'True' will profile the model training

    @property
    def dataset_config(self):
        if self.dataset_type is not None:
            return create_config_class_from_args(
                DataArgs.registrable_class.get_config_class_by_name(self.dataset_type),
                self,
            )
        else:
            raise ValueError(
                "Trying to access dataset config without specifying `dataset_type`!"
            )

    def __post_init__(self):
        if not self.compute_strategy:
            self.compute_strategy = "auto"

        if self.model is not None and self.model_family is None:
            # attempt to infer the model family from the model name
            if "t5" in self.model or "T0" in self.model:
                self.model_family = "seq2seq"
            else:
                self.model_family = "gpt"
            logger.warning(
                "Model family not specified, assuming {}".format(self.model_family)
            )

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

        if self.train_batch_size % self.micro_batch_size != 0:
            raise ValueError(
                "The training batch size must be divisible by the micro batch size."
            )

        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size

        if self.finetune_task_path is not None:
            if (
                not os.path.exists(self.finetune_task_path)
                and self.finetune_task_name is None
            ):
                raise ValueError(f"Task path {self.finetune_task_path} does not exist!")

            # resolve task keys
            task_names = []
            meta_tasks = self.finetune_task_name.split(",")
            # load task names from json file
            with open(self.finetune_task_path) as f:
                task_sets = json.load(f)

            for task_name in meta_tasks:
                # try to fetch task_names from the file
                if task_name in task_sets:
                    task_names.extend(task_sets[task_name])
                else:
                    task_names.append(task_name)

            self.finetune_task_name = ",".join(task_names)
            self.finetune_task_path = None

        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            warn_once(
                "You have multiple GPUs, but your device count is not being taken "
                + "into account when computing `gradient_accumulation_steps`."
            )

    def to_hf_training_args(self) -> "TrainingArguments":
        from transformers import TrainingArguments

        return TrainingArguments(
            run_name=self.wandb_run_name
            or self.expert_name
            or str(self.finetune_task_name),
            use_cpu=self.compute_strategy == "cpu",
            overwrite_output_dir=True,
            output_dir=self.output_dir,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.predict_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=1,
            bf16=self.precision == "bf16",
            fp16=self.precision == 16,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            load_best_model_at_end=True,
            warmup_steps=self.warmup_steps if self.warmup_steps > 0 else 0,
            warmup_ratio=self.warmup_proportion if self.warmup_proportion > 0 else 0,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.total_steps,
            save_total_limit=1,
            remove_unused_columns=False,
            save_strategy="epoch" if not self.save_every else "steps",
            eval_strategy="epoch" if not self.eval_every else "steps",
            save_steps=self.save_every,
            eval_steps=self.eval_every,
            ddp_find_unused_parameters=False,
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
    """
    Multi-expert configuration class that allows setting selectors and modifiers.
    In the future, we can remove the modifier support from this configuration, for now we leave it
    as it simplifies current tests and experiments.
    """

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
    es_metric: str = "loss"
    n_ng_iterations: int = 30  # number of iterations for LoraHub
    recompute_prototypes: bool = False


@dataclass
class MoEExpertConfig(MultiExpertConfig):
    moe_ent_reg: float = 0.0
    moe_ent_free_bits: float = 0.0
    moe_num_experts: int = 8
    init_from_scratch: bool = True
    pk_use_batchnorm: bool = True
    down_proj_layer: str = (
        "fc1"  # this is for the PEER container, it signals the names of the down and up projecting layers
    )
    up_proj_layer: str = (
        "fc2"  # this is for the PEER container, it signals the names of the down and up projecting layers
    )


@dataclass
class RankerConfig(TrainingArgs, SelectorArgs):
    encoder_model_name: str = "all-MiniLM-L6-v2"
    text_embedding_dim: int = 384
    expert_embedding_dim: int = 512
    projection_dim: int = 512
    val_check_interval = 1.0
    limit_val_batches: float = 1.0
    limit_train_batches: float = 1.0
