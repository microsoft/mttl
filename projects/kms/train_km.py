import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from nqa_datamodule import NQADatamodule  # noqa: F401
from utils.callbacks import LogMttlArgs

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


class DCDTrainer(ExpertModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.mttl_args.dcd_logit_loss or self.mttl_args.dcd_hidden_state_loss
        self.kl_loss = torch.nn.KLDivLoss(reduction="none")

    def get_reference_model(self, model):
        model = get_original_model(model)
        model.disable_modifiers()
        return model

    def get_student_model(self, model):
        model = get_original_model(model)
        model.enable_modifiers()
        return model

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
            ref_model = self.get_reference_model(model)

            outputs = ref_model(
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

            """
            # n_layers, seq_len, D
            target_hidden_states = torch.stack(target_hidden_states)

            # note that the last hidden state is the output to the lm_head, so we can remove
            target_hidden_states = target_hidden_states[:-1, ...]

            target_hidden_states = model.model.model.norm(target_hidden_states).to(model.model.lm_head.weight.dtype)
            target_hidden_logits = model.model.lm_head(target_hidden_states)

            target_hidden_logits = torch.cat(
                (target_hidden_logits, target_logits.unsqueeze(0)), dim=0
            )
            
            breakpoint()
            xx = 1
            """

        # for the context-less pass, we need to enable the adapter
        stu_model = self.get_student_model(model)

        outputs = stu_model(
            input_ids=nc_input_ids,
            attention_mask=nc_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        loss = 0.0
        if self.mttl_args.dcd_hidden_state_loss:
            losses = []
            for layer_id, (actual_states, target_states) in enumerate(
                zip(outputs.hidden_states, target_hidden_states)
            ):
                actual_states = actual_states[nc_labels != -100, :]

                if actual_states.size(0) != target_states.size(0):
                    # this shouldn't happen, but sometimes it does probably due to weird tokenization issues
                    logger.warning("Skipping batch due to mismatch in shape")
                    continue

                # Loss is the mean abs difference between target and predicted states,
                # normalised by mean magnitude of target states
                loss = (
                    actual_states - target_states
                ).abs().mean() / target_states.abs().mean()
                losses.append(loss)

            # Can we log the `kl_loss` of different layers ?
            if len(losses) == 0:
                # this happens when shape mismatch due to tokenization issues, should happen rarely
                fake_loss = actual_states.sum() * 0.0
                return (fake_loss, outputs.logits) if return_outputs else fake_loss

            loss = torch.mean(torch.stack(losses))

        if self.mttl_args.dcd_logit_loss:
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
    # the following are DCD Trainer arguments
    dcd_logit_loss: bool = True
    dcd_hidden_state_loss: bool = True
    dcd_hidden_logit_loss: bool = False


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

    if args.library_id:
        expert_library = create_library(args)

        if model_config.expert_name in expert_library:
            # gracefully exit if expert already exists in library
            logger.warning(
                f"Expert {model_config.expert_name} already exists in library!"
            )
            return

    model = ExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    )

    # Not all argument are being logged. This remedies it
    callbacks = [LogMttlArgs(training_args)]

    if training_args.nqa_dataset is not None:
        # load the NQA callback to monitor zero-shot performance
        from nqa_callback import NQAZeroShotCallback

        data_args = copy.deepcopy(training_args)
        data_args.dataset = training_args.nqa_dataset
        data_args.dataset_type = "narrativeqa"
        callback = NQAZeroShotCallback(model, data_args)
        callbacks.append(callback)

    if training_args.loss_function == "dcd":
        trainer: DCDTrainer = DCDTrainer(
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

    trainer.save_model(args.output_dir + "/best_model")
    trainer.save_state()

    # Maybe save to Expert Library
    if args.library_id:
        upload_library(
            expert_library,
            best_model_path or model,
            expert_name=model_config.expert_name,
        )


if __name__ == "__main__":
    args = KMArguments.parse()

    assert not args.dcd_hidden_logit_loss, "Not supported yet"

    train_km(args)
