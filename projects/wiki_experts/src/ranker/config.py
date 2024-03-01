from mttl.models.expert_config import ExpertConfig


class RankerConfig(ExpertConfig):
    def _set_defaults(self):
        super()._set_defaults()
        # training expert
        self.ranker_model = "classifier"
        self.ranker_path = None

    def post_init(self, silent=False):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size
