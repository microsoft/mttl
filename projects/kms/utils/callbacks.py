import wandb
from transformers import TrainerCallback


class LogMttlArgs(TrainerCallback):
    def __init__(self, additional_args):
        super().__init__()
        self.additional_args = additional_args

    def on_train_begin(self, args, state, control, **kwargs):
        # Log any additional arguments at the start of trainixg
        wandb.config.update(self.additional_args, allow_val_change=True)
