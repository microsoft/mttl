import os

from mttl.config import Config
import mttl.datamodule.flan_tasks


class ExpertConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.load_in_8bit = False
        self.wandb_project = None
        self.tensorboard = False
        self.hf_token_hub = None
        self.hf_lib_id = None
        self.hf_repo_id = None
        self.run_name = "default"

        # just a lame flag to 0 out all adapter weights
        self.baseline = False
        # sparsify adapter weights to this sparsity level
        self.sparsity = 0.0
        # only use a very small portion of the available experts
        self.subsample_library_experts = 0
        # rank / retrieve top k experts
        self.ranker_top_k = 1
        self.ranker_path = None
        self.ranker_model = None

        self.expert_name = None
        self.routing = "subject"
        self.mmlu_test_split = "test"
        self.load_module = None
        self.module_graph = None
        self.micro_batch_size = None
        self.validation_portion = 0.03

        self.source_template = None
        self.augment_few_shot = 0

        self.expand_val_set_w_downstream = False

        self.eval_mmlu_callbacks_every = 0
        self.eval_test_set_callback_every = 0
        self.eval_rougeL_callback_every = 0
        self.test_sets_callbacks = []

        self.use_custom_valid_callback = False  # if True use custom callback to early top on eval loss  instead of lightning callback

        self.data_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")

        self.mmlu_use_hard_prompt = None
        self.eval_mmlu_few_shot = True  # use few-shot for mmlu, default
        self.eval_mmlu_flag = False
        self.eval_metric = "loss"
        self.use_vllm = False

    def post_init(self):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size

        if self.finetune_task_name is not None:
            if self.finetune_task_name in mttl.datamodule.flan_tasks.__dict__.keys():
                self.finetune_task_name = getattr(
                    mttl.datamodule.flan_tasks,
                    self.finetune_task_name,
                    self.finetune_task_name,
                )
