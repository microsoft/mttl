from mttl.config import Config
import os


class RoutingConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.rank = 1
        self.load_dtype = "float32"
        self.prune_unused_loras = True
        self.init_b_random = False
        self.lora_dropout = 0
        self.lora_alpha = 16
        self.same_lora_init = 0
        self.load_in_8bit = False
        self.micro_batch_size = 4
        self.share_lora_at_attn = 0
        self.share_lora_a = False
        self.merge_A_B_seperately = True
        self.train_on_inputs = False
        self.padding_side = "right"
        self.adapter_modules = None
        self.router_selector_use_distances = False
        self.adapter_layers = 0  # llama adapter
        self.adapter_len = 0  # llama adapter
        self.use_4_bit_backbone = False
        self.wandb_project = None
        self.switch_to_average = 0

        self.router_weight_decay = None
        self.param_names_added_to_sd = ""  # define additional params that will be added to state dict additionally to the trainable ones.
        self.predict_cluster = None  # topic or skill
        self.dst_dir = None  # dir of jsonl dataset

        self.fast_dev_run = False
        self.fast_debug_run = False

        self.hf_token_hub = None
        self.eval_ds_limit = 1
        self.train_only_cluster = None
        self.validation_portion = 0.03
        self.per_cluster_test = False
        self.use_test_set = False  # wether to use examples marked as is_test = 1 in ClusterInfo as test set
        self.aux_mi_loss_factor = 1

        # XRouter
        self.xrouter_load_balancing = False
        self.xrouter_x_cond = True
        self.xrouting_option = 0  # only applies to x_router routing, depreciated
        self.xrouter_normalize_weights = False
        self.xrouter_normalize_input = False
        self.xrouter_reverse_kl = False
        self.xrouter_normal_innit = True
        self.xrouter_use_attn = False
        self.xrouter_sim_metric = "kl"
        self.xrouting_sep_teacher_student = False
        self.xrouter_init_scale = 0.02
        self.xrouter_x4target_detach = True
        self.xr4_option = None  # "switch" #, "default", "teacher_output"
        self.xrouter_x4_target = "posterior"  # , "prior" -- wich router logits to use for x4 at trainign time.

        self.router_learning_rate = None
        self.eval_hellaswag = True
        self.eval_arc = True
        self.eval_truthfulqa = True
        self.eval_superni = True
        self.eval_mmlu = True
        self.eval_batches = 50
        self.gen_alpaca_eval = False

        self.data_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")

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
