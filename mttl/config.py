import argparse
import ast
import json
import os
from string import Template
from typing import Dict

from mttl.utils import logger, setup_logging


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
        pass

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
        self.cache_dir = os.getenv("CACHE_DIR", "./cache")

        # Data config
        self.dataset = None
        self.custom_tasks_splits = None

        self.data_dir = os.getenv("TRAIN_DIR", "/tmp/")
        self.output_dir = os.getenv("OUTPUT_DIR", "./output")

        self.finetune_task_name = None
        self.example_to_ids_path = None  # path to clustering of data
        self.embeddings_path = None

        # NI related configs
        self.use_task_descriptions = False  # Use task descriptions
        self.max_num_instances_per_task = (
            100  # Max instances per training task (applies to NI)
        )
        self.num_pos_examples = (
            0  # Use some few-shot examples if possible (applies to NI)
        )

        self.task_prefix = None  # xfit has task prefixes detailing # of shots, seed, etc; this is automatically filled in at fine-tuning time
        self.exp_name = None
        self.wandb_project = None
        self.padding_side = "right"
        self.truncation_side = "right"
        self.max_input_length = 512
        self.max_output_length = 64
        self.num_beams = 4
        self.append_another_bos = False
        self.do_lowercase = False
        self.freeze_embeds = False

        # T0 related configs
        self.use_t0_templates_as_tasks = (
            False  # if True, then t0 consists of 313 tasks, otherwise 38
        )
        self.use_t0_few_shot_training_set = False  # if True, then use 100 examples per task during training + 100 examples per validation task

        # Filtering configs, useful for flan flat, etc.
        self.include_template_type = "zs_noopt"
        self.include_task_source = "P3,Flan2021,CoT"
        self.remove_phi_eval_tasks = False

        # Training config
        self.compute_strategy = None
        self.padding_side = "right"
        self.scheduler = "linear_decay_with_warmup"
        self.checkpoint = None  # load from checkpoint
        self.checkpoint_step = None  # load from checkpoint in format of global_stepX.pt
        self.backbone_checkpoint = None  # load the backbone from here
        self.train_batch_size = 8
        self.predict_batch_size = 32
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

        self.subsample_train = None
        self.subsample_dev = None
        self.subsample_test = None
        self.subsample_per_task = False

        self.ni_online_eval = False  # zero-shot online eval for ni
        self.t0_online_eval = False  # zero-shot eval for t0
        self.early_stop_on_zero_shot = False  # zero-shot early stopping

        # auxiliary losses
        self.ortho_loss = 0.0  # orthogonality between skills
        self.task_loss = 0.0  # task prediction loss (mi between tasks and skills)
        self.l1_loss = 0.0  # sparsity of the logits
        self.mi_loss = (
            0.0  # mi between tasks and skills (difference of entropies method)
        )
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

        self.precision = "32"
        self.monitor_grad_alignment_on = None

        self.model_modifier = None
        self.adapter_type = None

        self.lora_rank = 16
        self.lora_dropout = 0.0
        self.lora_init_scale = 0.01
        self.lora_alpha = 1.0
        self.lora_warmup = False
        self.lora_init_b_random = False
        self.lora_dropout = 0.0

        # n-skills for router-based methods
        self.n_skills = 8
        self.n_tasks = None
        self.task_names = None

        # which modules to modify and which layers for adapters
        self.modify_modules = None
        self.modify_layers = None
        self.tie_params = None  #  "q_proj\\.lora_a|k_proj\\.lora_a|v_proj\\.lora_a" to share lora_a for q,k,v

        """
        router_granularity : how granular is the module selection
        coarsegrained : 1 single selector across all linear layers
        coderwise : 2 selectors (1 for encoder, 1 for decoder)
        blockwise : 1 selector for each block of K attention layers (and layernorm)
        layerwise : 1 selector for each attention layer (and layernorm)
        finegrained : 1 selector for every linear layer
        """
        self.router_granularity = "finegrained"  # router granularity
        self.router_selector = None  # router selector
        self.router_weight_decay = None  # router weight decay
        self.router_learning_rate = None

        # Polytropon related hyper-parameters
        self.n_splits = 1  # number of splits for poly-s
        self.router_selector_cluster_temp = 1.0  # temperature for the cluster selector
        self.poly_average_correction = False  # correct the poly average
        self.poly_use_shared_skill = False  # use one skill shared by all tasks
        self.skip_unseen_tokens = (
            True  # skip unseen tokens in PerTokenPoly during evaluation
        )

        self.module_logits_relaxed_bernoulli = True
        self.module_logits_straight_through = False
        self.module_logits_learning_rate = 0.1
        self.adapters_learning_rate = None
        self.adapters_weight_decay = None
        self.module_logits_dropout = 0.0
        self.module_logits_l2_norm = False

        self.augment_mmlu: bool = False

        # soft prompts
        self.soft_prompt_length: int = 10
        self.patch_last_k_layers: int = -1
        self.soft_prompt_mlp_dim: int = None
        self.soft_prompt_hidden_dim: int = None
        self.soft_prompt_learn_kv: bool = False
        self.prompt_placement: str = "prefix"
        self.add_routing_token: bool = False
