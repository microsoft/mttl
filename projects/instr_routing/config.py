from mttl.config import Config
import os


class RoutingConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.merge_after_op = False
        self.micro_batch_size = 4
        self.load_in_8bit = False

        self.wandb_project = None
        self.tensorboard = False
        self.switch_to_average = 0

        # scale the output a bit
        self.lora_alpha = 16

        self.router_weight_decay = None  # weight decay for the routing parameters
        self.router_learning_rate = None  # learning rate of the routing parameters
        self.router_temperature = 1.0  # temperature of router for softmax

        self.router_teacher_temperature = (
            1.0  # temperature of router for teacher softmax
        )
        self.router_normalize_weights = (
            False  # l2 normalize cluster centroids before routing
        )

        self.router_kl_factor = (
            1.0  # factor for the posterior entropy term in the vsmear router
        )
        self.router_kl_func = "kl"
        self.router_center_momentum = (
            0.0  # centering momentum a-la DINO_v2, if 0. don't use centering
        )
        self.router_shared_weights = True  # share weights between teacher and student

        self.fast_dev_run = False
        self.remote_token = None
        self.validation_portion = 0.03

        self.eval_superni = False
        self.eval_mmlu = False
        self.eval_batches = -1
        self.eval_avg = True
        self.validate_after_training = True
        self.eval_in_8bit = False

        self.data_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")
        # logging
        self.selector_log_per_layer = True
        self.mmlu_callback = True
        # softmoe
        self.use_causal_mask_for_D = True

        # smear
        self.smear_gaussian_init = False

        # vsmear_x4
        self.xrouter_x4_target = "prior"
        self.xrouter_x4target_detach = True

        # task vsmear
        self.task_vsmear_detach_prior_input = False
        self.task_vsmear_aux_lambda = 1.0

        # soft prompts
        self.soft_prompt_length = 10
        self.patch_last_k_layers = -1
        self.soft_prompt_mlp_dim = None
        self.soft_prompt_hidden_dim = None
        self.soft_prompt_learn_kv = False

    def post_init(self):
        if self.eval_mmlu and "MMLU_DATA_DIR" not in os.environ:
            raise ValueError("MMLU_DATA_DIR not set in env but eval_mmlu = True.")

        if self.eval_superni and "NI_DATA_DIR" not in os.environ:
            raise ValueError("NI_DATA_DIR not set in env but eval_superni = True.")

        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size
