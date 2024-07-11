import os
import re
from collections import defaultdict
from enum import Enum
from typing import Any, Mapping

import torch
from torch import nn
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer import Trainer
from transformers.utils import PushToHubMixin

from mttl.models.get_scheduler import get_scheduler_with_args
from mttl.utils import logger


def transfer_batch_to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


class EfficientCheckpointHfModel(nn.Module, PushToHubMixin):
    _keys_to_ignore_on_save = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save_pretrained(self, save_directory):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(
                "Provided path ({}) should be a directory, not a file".format(
                    save_directory
                )
            )
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if k not in self._keys_to_ignore_on_save
            }

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        model_to_save.config.save_pretrained(save_directory)
        torch.save(state_dict, output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))

    def state_dict(self):
        state_dict = {}

        for k, p in self.named_parameters():
            # we can safely avoid dumping this parameter if it is both
            # not in the trainable parameters and was not loaded from checkpoint
            if p.requires_grad:
                state_dict[k] = p.data

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        return super().load_state_dict(state_dict, strict=False)


class MttlTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        from transformers.trainer import ShardedDDPOption
        from transformers.utils import is_sagemaker_mp_enabled

        from mttl.models.get_optimizer import get_optimizer
        from mttl.utils import logger

        if is_sagemaker_mp_enabled():
            raise NotImplementedError(
                "SageMaker Model Parallelism is not yet supported in our Trainer."
            )

        opt_model = self.model

        if self.optimizer is None:
            param_groups = get_optimizer(self.args)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                raise NotImplementedError(
                    "Sharded DDP is not yet supported in our Trainer."
                )
            else:
                self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum(
                                {
                                    p.data_ptr(): p.numel() for p in module.parameters()
                                }.values()
                            )
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(
                                module, "weight", {"optim_bits": 32}
                            )
                            logger.debug(
                                f"bitsandbytes: will optimize {module} in fp32"
                            )
                    logger.info(f"skipped: {skipped/2**20}M params")
        return self.optimizer

    def get_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        self.lr_scheduler = get_scheduler_with_args(
            optimizer,
            lr_scheduler_type=self.args.scheduler,
            warmup_steps=self.args.get_warmup_steps(num_training_steps),
            total_steps=num_training_steps,
            learning_rate=self.args.learning_rate,
        )
        return self.lr_scheduler
