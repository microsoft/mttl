from mttl.config import Config
import os


class MHRConfig(Config):
    def _set_defaults(self):
        self.cache_dir = os.getenv("CACHE_DIR", "./cache")
        self.free_up_space = False
        # Data config
        self.dataset = None
        self.custom_tasks_splits = None
        self.train_dir = os.getenv("AMLT_DATA_DIR", "/tmp/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "./output")
        self.finetune_task_name = None
        self.example_to_ids_path = None  # path to clustering of data
        self.embeddings_path = None
        self.use_task_descriptions = False  # Use task descriptions
        self.num_pos_examples = 0  # Use some few-shot examples if possible (applies to NI)
        self.task_prefix = None    # xfit has task prefixes detailing # of shots, seed, etc; this is automatically filled in at fine-tuning time
        self.exp_name = None
        self.wandb_project = None
        self.max_input_length = 512
        self.max_output_length = 64
        self.num_beams = 4
        self.append_another_bos = False
        self.do_lowercase = False
        self.freeze_embeds = False

        self.use_t0_templates_as_tasks = False     # if True, then t0 consists of 313 tasks, otherwise 38
        self.use_t0_few_shot_training_set = False  # if True, then use 100 examples per task during training + 100 examples per validation task

        # Training config
        self.compute_strategy = None
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
        self.debug = False
        self.seed = 42

        self.ni_online_eval = False   # zero-shot online eval for ni
        self.t0_online_eval = False   # zero-shot eval for t0
        self.early_stop_on_zero_shot = False  # zero-shot early stopping

        # auxiliary losses
        self.ortho_loss = 0.          # orthogonality between skills
        self.task_loss = 0.           # task prediction loss (mi between tasks and skills)
        self.l1_loss = 0.             # sparsity of the logits
        self.mi_loss = 0.             # mi between tasks and skills (difference of entropies method)
        self.mc_loss = 0.             # T-Few
        self.length_norm = 0.         # T-Few
        self.unlikely_loss = 0.       # T-Few
        self.poly_unlikely_loss = 0.  # poly unlikelihood loss
        self.finetune_type = None     # ["F", "A", "Z", "MuZ", "Poly", "PolyRand"]
        self.finetune_skip_es = False  # skip early stopping while fine-tuning
        self.finetune_use_last_checkpoint = False  # use always the best valid_perf checkpoint if available
        self.model = None
        self.precision = "32"
        self.monitor_grad_alignment_on = None

        self.model_modifier = None
        self.lora_randb_init = False
        self.lora_rank = 16
        self.lora_init_scale = 0.01
        self.lora_scaling_rank = 0
        self.lora_kaiming_init = False
        self.lora_warmup = False
        self.lora_modules = self.patch_modules = None
        self.lora_layers = self.patch_layers = None
        self.n_skills = 8
        self.n_tasks = None

        # Polytropon related hyper-parameters
        self.n_splits = 1                      # number of splits for poly-s
        self.poly_selector = "poly"            # poly, poly_cluster
        self.poly_selector_cluster_temp = 1.0  # temperature for the cluster selector
        self.poly_average_correction = False   # correct the poly average
        self.poly_use_shared_skill = False     # use one skill shared by all tasks

        """
        poly_granularity : how granular is the module selection : 
        coarsegrained : 1 single selector across all linear layers
        coderwise : 2 selectors (1 for encoder, 1 for decoder)
        blockwise : 1 selector for each block of K attention layers (and layernorm)
        layerwise : 1 selector for each attention layer (and layernorm) 
        finegrained : 1 selector for every linear layer 
        """
        self.poly_granularity = 'finegrained'

        self.module_logits_relaxed_bernoulli = True
        self.module_logits_straight_through = False
        self.module_logits_learning_rate = 0.1
        self.adapters_learning_rate = None
        self.adapters_weight_decay = None
        self.module_logits_dropout = 0.
        self.module_logits_l2_norm = False
