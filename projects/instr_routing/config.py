from mttl.config import Config
import os


class RoutingConfig(Config):
    def _set_defaults(self):
        super()._set_defaults()

        self.load_in_8bit = False
        self.micro_batch_size = 4

        self.wandb_project = None
        self.switch_to_average = 0

        self.router_weight_decay = None
        self.router_learning_rate = None
        self.fast_dev_run = False

        self.hf_token_hub = None
        self.validation_portion = 0.03

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
