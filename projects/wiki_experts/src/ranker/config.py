from projects.wiki_experts.src.config import ExpertConfig


class RankerConfig(ExpertConfig):
    def _set_defaults(self):
        super()._set_defaults()
        # training expert
        self.ranker_model = "classifier"
        self.ranker_path = None
        self.subsample = -1
        self.encoder_model_name = "all-MiniLM-L6-v2"
        self.text_embedding_dim = 384
        self.expert_embedding_dim = 512
        self.projection_dim = 512

    def post_init(self):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.train_batch_size

        # to reproduce setup in https://github.com/daanelson/alpaca-lora
        self.gradient_accumulation_steps = (
            self.train_batch_size // self.micro_batch_size
        )
        self.train_batch_size = self.micro_batch_size
