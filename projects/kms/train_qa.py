import copy
import logging
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, upload_file
from lightning_fabric import seed_everything

# register this datamodule!
from nqa_datamodule import NQADatamodule

from mttl.arguments import MultiExpertConfig
from mttl.logging import setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig, MoEModelConfig
from mttl.models.hf.trainer import LMTrainer
from mttl.models.km_model import KMMoEModel, KMMoEModelConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import create_library, remote_login, upload_library

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class KEArguments(MultiExpertConfig):
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = "sordonia/narrativeqa"
    ke_hf_path: str = None


def train_ke(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    if training_args.library_id:
        logger.info("Loading expert library: %s", training_args.library_id)
        if training_args.router_granularity != "coarsegrained":
            logger.warning("Overwriting `router_granularity` to 'coarsegrained'")
            training_args.router_granularity = "coarsegrained"

        if training_args.router_selector != "ke_selector":
            logger.warning("Overwriting `router_selector` to 'ke_selector'")
            training_args.router_selector = "ke_selector"

        # expert_library = create_library(training_args)
        model_config = KMMoEModelConfig(
            base_model=training_args.model,
            library_id=training_args.library_id,
            selector_config=training_args.selector_config,
        )
        model = KMMoEModel(model_config)
        if model.ke_expert_name not in training_args.trainable_param_names:
            # Let's provide a fix that works for the current setup
            if training_args.trainable_param_names == ".*lora_[ab].*":
                logger.warning("Overwriting `trainable_param_names` to include the KE")
                training_args.trainable_param_names = (
                    f".*lora_[ab].{model.ke_expert_name}.*"
                )
            else:
                raise ValueError(
                    "Please ensure that the Knowledge Extractor will be trained"
                )

        # Make sure that when creating the datamodule, we only load tasks
        # for which we have trained KM experts
        if training_args.finetune_task_name:
            logger.warning("Overwriting `finetune_task_name` to match the KM experts")

        training_args.finetune_task_name = list(
            filter(lambda x: x != model.ke_expert_name, model.experts_names)
        )
    else:
        logger.info("Loading model without expert library")
        model_config = ExpertModelConfig(
            base_model=args.model,
            expert_name=args.expert_name or "KE",
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

    trainer = LMTrainer(model=model, args=training_args, callbacks=callbacks)
    trainer.train()

    # Get the best checkpoint
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        logger.info("Best model checkpoint: %s", best_model_path)
        model = type(model).from_pretrained(best_model_path)

    # Maybe save to Expert Library
    if args.ke_hf_path:
        # TODO: make sure that pushing expert in MoE works
        if isinstance(model, KMMoEModel):
            ke_expert = model.get_expert_instance(model.ke_expert_name)
            # creat a library and upload that expert
            lib_path, exp_name = args.ke_hf_path.rsplit("/", 1)
            expert_library = ExpertLibrary.get_expert_library(lib_path, create=True)
            expert_library.add_expert(ke_expert, exp_name)
        else:
            model.push_to_hub(args.ke_hf_path)


if __name__ == "__main__":
    args = KEArguments.parse()
    assert args.dataset_config
    import json

    train_ke(args)
