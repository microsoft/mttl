from mttl.config import Config
import os


class ExpertConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.load_in_8bit = False
        self.wandb_project = None
        self.tensorboard = False
        self.hf_token_hub = None
        self.hf_repo_id = None

        self.expert_name = None
        self.routing = "subject"
        self.mmlu_test_split = "test"
        self.load_module = None
        self.module_graph = None
        self.micro_batch_size = None
        self.validation_portion = 0.03

        self.expand_val_set_w_downstream = False

        self.n_ng_iterations = 2
        self.use_vllm = False
        self.expert_routing = None

        self.data_dir = os.getenv("AMLT_DATA_DIR", "~/data/")
        self.output_dir = os.getenv("AMLT_OUTPUT_DIR", "tmp/instruction_learning/")

        # training expert
        self.eval_mmlu_flag = False

    def post_init(self):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size
