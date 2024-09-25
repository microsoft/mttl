import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything

from mttl.arguments import ExpertConfig
from mttl.logging import setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.hf.trainer import ExpertModelTrainer, LMTrainer
from mttl.utils import create_library, remote_login, upload_library

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_original_model(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


class DeepContextDistillationTrainer(ExpertModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

    def compute_loss(self, model, inputs, return_outputs=False):
        # document + small task prompt + task output (e.g. summary, or question and answer)
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        attention_mask = inputs["attention_mask"]

        # small task prompt + task output (e.g. summary, or question and answer)
        nc_input_ids = inputs["nc_input_ids"]
        nc_labels = inputs["nc_labels"]
        nc_attention_mask = inputs["nc_attention_mask"]

        with torch.no_grad():
            # for the context-aware pass, we need to disable the adapter
            get_original_model(model).disable_adapter()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            target_hidden_states = [
                hidden_state[labels != -100, ...]
                for hidden_state in outputs.hidden_states
            ]
            target_logits = outputs.logits[labels != -100, :]

        # for the context-less pass, we need to enable the adapter
        get_original_model(model).enable_adapter()

        outputs = model(
            input_ids=nc_input_ids,
            attention_mask=nc_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        losses = []
        for actual_states, target_states in zip(
            outputs.hidden_states, target_hidden_states
        ):
            actual_states = actual_states[nc_labels != -100, :]

            # Calculate mean magnitude of target states
            mean = torch.sum(torch.abs(target_states)) / (actual_states.size(-1))

            losses.append(
                # Loss is the mean abs difference between target and predicted states,
                # normalised by mean magnitude of target states
                torch.sum(torch.abs(actual_states - target_states))
                / (mean * np.prod(target_states.shape))
            )

        loss = torch.mean(torch.stack(losses))

        # Add KL divergence between target and predicted output distributions to loss
        target_probs = F.softmax(target_logits, dim=-1)
        preds = F.log_softmax(outputs.logits[nc_labels != -100, ...], dim=-1)
        kl_loss = self.kl_loss(preds, target_probs).sum(dim=-1).mean()
        loss = loss + kl_loss

        return (loss, outputs.logits) if return_outputs else loss


@dataclass
class KMArguments(ExpertConfig):
    loss_function: str = "dcd"
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = "sordonia/narrativeqa"


def train_km(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    model_config = ExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name or args.finetune_task_name,
        modifier_config=args.modifier_config,
    )

    model = ExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    )

    callbacks = []
    if training_args.nqa_dataset is not None:
        # load the NQA callback to monitor zero-shot performance
        from nqa_callback import NQAZeroShotCallback

        data_args = copy.deepcopy(training_args)
        data_args.dataset = training_args.nqa_dataset
        callback = NQAZeroShotCallback(model, data_args)
        callbacks.append(callback)

    if training_args.loss_function == "dcd":
        trainer: DeepContextDistillationTrainer = DeepContextDistillationTrainer(
            model=model,
            args=training_args,
            callbacks=callbacks,
        )
    elif training_args.loss_function == "lm":
        trainer: LMTrainer = LMTrainer(
            model=model,
            args=training_args,
            callbacks=callbacks,
        )
    else:
        raise ValueError(f"Unsupported loss function: {training_args.loss_function}")

    trainer.train()

    # Get the best checkpoint
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        logger.info("Best model checkpoint: %s", best_model_path)

    # Maybe save to Expert Library
    if args.library_id:
        expert_library = create_library(args)
        upload_library(
            expert_library,
            best_model_path or model,
            expert_name=args.finetune_task_name,
        )


if __name__ == "__main__":
    args = KMArguments.parse()
    train_km(args)
