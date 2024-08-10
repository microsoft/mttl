import argparse
import ast
import json
import os
from dataclasses import fields
from string import Template
from typing import Dict

from mttl.logging import logger, setup_logging


class Config:
    def __init__(self, filenames=None, kwargs=None, raise_error=True, silent=False):
        # Stores personalization of the config file in a dict (json serializable)

        self._updated_kwargs = {}
        self.filenames = filenames
        self._set_defaults()

        overwrite_logs = []
        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(
                        os.getenv("CONFIG_PATH", default="configs"), filename
                    )

                if not os.path.exists(filename) and ".json" not in filename:
                    filename = filename + ".json"

                overwrite_logs += self.update_kwargs(
                    json.load(open(filename)),
                    eval=False,
                    raise_error=raise_error,
                    silent=silent,
                )

        if kwargs:
            overwrite_logs += self.update_kwargs(
                kwargs, raise_error=raise_error, silent=silent
            )

        # setup logging to the output dir
        try:
            setup_logging(self.output_dir)
        except Exception as e:
            if raise_error:
                raise ValueError("Error setting up logging") from e
            elif not silent:
                logger.warning(f"Error setting up logging to {self.output_dir}")

        # log the overwrites
        for log in overwrite_logs:
            logger.warning(log)

        self.post_init(silent=silent)

    def post_init(self, silent=False):
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

    @classmethod
    def fromdict(cls, data):
        _ = data.pop("_updated_kwargs", None)
        return cls(kwargs=data, raise_error=False, silent=True)

    def asdict(self) -> Dict:
        from mttl.models.utils import convert_hps_to_dict

        return convert_hps_to_dict(self.__dict__)

    def was_overridden(self, key):
        return key in self._updated_kwargs

    def was_default(self, key):
        return key not in self._updated_kwargs

    def update_kwargs(self, kwargs, eval=True, raise_error=True, silent=False):
        overwrites_log = []
        for k, v in kwargs.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    v = v
            else:
                v = v

            if not hasattr(self, k) and raise_error:
                raise ValueError(f"{k} is not in the config")

            if eval and not silent:
                overwrites_log.append(f"Overwriting {k} to {v}")

            if type(v) == str and "$" in v:
                # this raises an error if the env. var does not exist
                v = Template(v).substitute(os.environ)

            setattr(self, k, v)
            self._updated_kwargs[k] = v
        return overwrites_log

    def __getitem__(self, item):
        return getattr(self, item, None)

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        import copy

        to_save = copy.deepcopy(self.__dict__)
        to_save.pop("_updated_kwargs")

        return json.dumps(to_save, indent=4, sort_keys=False)

    def save_config(self, output_dir):
        """
        Saves the config
        """
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.json"), "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")

    @classmethod
    def parse(
        cls,
        extra_kwargs=None,
        raise_error=True,
        parent=None,
        return_parser=False,
        c=None,
    ):
        import itertools

        # dont do it if called from jupyter notebook
        if c is None:
            parser = (
                argparse.ArgumentParser(parents=[parent])
                if parent
                else argparse.ArgumentParser()
            )
            parser.add_argument("-c", "--config_files", required=False)
            parser.add_argument("-k", "--kwargs", nargs="*", action="append")
            args = parser.parse_args()
        else:
            args = argparse.Namespace()
            args.kwargs = None
            args.config_files = c

        kwargs = {}
        if args.kwargs:
            kwargs_opts = list(itertools.chain(*args.kwargs))
            for value in kwargs_opts:
                key, _, value = value.partition("=")

                # allows multiple values for a given option when specified in the command line!
                if key in kwargs:
                    if type(kwargs[key]) != list:
                        kwargs[key] = [kwargs[key]]
                    kwargs[key].append(value)
                else:
                    kwargs[key] = value

        args.kwargs = kwargs
        if extra_kwargs:
            args.kwargs.update(extra_kwargs)

        config = cls(
            filenames=args.config_files, kwargs=args.kwargs, raise_error=raise_error
        )

        if return_parser:
            return config, args
        return config

    def _set_defaults(self):
        # call _set_defaults of all the parent classes
        for c in reversed(self.__class__.__mro__):
            if c == Config:
                continue
            c._set_defaults(self)


class TrainingConfig(Config):
    def _set_defaults(self):
        self.cache_dir = os.getenv("CACHE_DIR", "./cache")
        self.data_dir = os.getenv("TRAIN_DIR", "/tmp/")
        self.output_dir = os.getenv("OUTPUT_DIR", "./output")

        self.finetune_task_name = None
        self.example_to_ids_path = None  # path to clustering of data
        self.exp_name = None
        self.wandb_project = None

        # Training config
        self.compute_strategy = None
        self.padding_side = "right"
        self.scheduler = "linear_decay_with_warmup"
        self.checkpoint = None  # load from checkpoint
        self.checkpoint_step = None  # load from checkpoint in format of global_stepX.pt
        self.backbone_checkpoint = None  # load the backbone from here

        self.learning_rate = 1e-3
        self.warmup_proportion = 0.06
        self.trainable_param_names = ".*"
        self.non_trainable_param_names = None
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 0.1
        self.gradient_accumulation_steps = 1
        self.optimizer = "adamw"
        self.adafactor_scale_parameter = True
        self.adafactor_warmup_init = False
        self.adafactor_relative_step = False
        self.num_train_epochs = -1
        self.warmup_steps = -1
        self.total_steps = -1
        self.num_tasks_per_batch = None
        self.save_every = None
        self.eval_every = None
        self.eval_every_n_epoch = 1
        self.debug = False
        self.seed = 42
        self.eval_before_training = True

        self.mc_loss = 0.0  # T-Few
        self.length_norm = 0.0  # T-Few
        self.unlikely_loss = 0.0  # T-Few
        self.poly_unlikely_loss = 0.0  # poly unlikelihood loss
        self.finetune_type = None  # ["F", "A", "Z", "MuZ", "Poly", "PolyRand"]
        self.finetune_skip_es = False  # skip early stopping while fine-tuning
        self.finetune_use_last_checkpoint = (
            False  # use always the best valid_perf checkpoint if available
        )

        self.model = None
        self.model_family = None  # model family, either "gpt" or "encdec"
        self.attn_implementation = None

        self.precision = "32"
        self.monitor_grad_alignment_on = None

        self.model_modifier = None
        self.router_selector = None  # router selector

        self.router_weight_decay = None  # router weight decay
        self.router_learning_rate = None
        self.module_logits_learning_rate = 0.1
        self.adapters_learning_rate = None
        self.adapters_weight_decay = None
        self.module_logits_dropout = 0.0
        self.module_logits_l2_norm = False

        self.augment_mmlu: bool = False


class DatasetConfig(Config):
    def _set_defaults(self):
        from mttl.datamodule.base import DefaultDataModule

        for c in DefaultDataModule.registered_configs():
            for f in fields(c):
                setattr(self, f.name, None)


class Modifier(Config):
    def _set_defaults(self):
        from mttl.models.modifiers import Modifier

        for c in Modifier.registered_configs():
            for f in fields(c):
                setattr(self, f.name, None)


class SelectorArgs(Config):
    def _set_defaults(self):
        from mttl.models.containers.selectors import Selector

        for c in Selector.registered_configs():
            for f in fields(c):
                setattr(self, f.name, None)


class TrainExpertsArgs(SelectorArgs, ModifierArgs, DatasetArgs, TrainingArgs):
    pass
