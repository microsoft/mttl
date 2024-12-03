import wandb
from transformers import TrainerCallback

from mttl.utils import logger


class LogMttlArgs(TrainerCallback):
    def __init__(self, additional_args):
        super().__init__()
        self.additional_args = additional_args

    def on_train_begin(self, args, state, control, **kwargs):
        # check if wandb was initialized
        if wandb.run is None:
            logger.warning(
                "Wandb is not initialized. Skipping logging of additional args."
            )
            return

        # Log any additional arguments at the start of trainixg
        wandb.config.update(self.additional_args, allow_val_change=True)
