import copy
import logging
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from nqa_datamodule import NQADatamodule
from train_km import get_original_model

from mttl.arguments import MultiExpertConfig
from mttl.logging import setup_logging
from mttl.models.base_model import BaseExpertModel
from mttl.models.expert_model import (
    BaseExpertModel,
    ExpertModel,
    ExpertModelConfig,
    MoEModelConfig,
    MultiExpertMixin,
)
from mttl.models.hf.trainer import LMTrainer
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import create_library, remote_login, upload_library

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class KMMoEModelConfig(MoEModelConfig):
    ke_expert_name: str = None
    ke_experts_prefix: str = "KE"


@BaseExpertModel.register("moe_km", config_cls=MoEModelConfig)
class KMMoEModel(BaseExpertModel, MultiExpertMixin):
    """MoeModel that can accomodate a Knowledge Extractor"""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.modifier_config = config.modifier_config
        expert_library = ExpertLibrary.get_expert_library(self.config.library_id)

        # Now, we may want to try and test multiple knowledge extractors on the same library.
        # To do so, we need to be able to not load previously trained ones
        expert_names = expert_library.keys()
        ke_experts = list(
            filter(lambda x: x.startswith(self.config.ke_experts_prefix), expert_names)
        )
        for expert in sorted(list(expert_library.keys())):
            if expert not in ke_experts:
                self.add_expert_instance(expert_library[expert], expert_name=expert)

        assert len(self.experts_names) > 0, "No experts found in the library."

        # also need to add an additional expert for the KE
        # we will use the `ExpertConfig` of the first expert
        an_expert = self.get_expert_instance(self.experts_names[0])
        self.ke_expert_name = (
            self.config.ke_expert_name
            or f"{self.config.ke_experts_prefix}_{len(ke_experts)}"
        )
        self.add_empty_expert(
            self.ke_expert_name, expert_config=an_expert.expert_config
        )
        logger.info("Added KE expert: %s", self.ke_expert_name)


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

        ke_name = model.ke_expert_name
        if ke_name not in training_args.trainable_param_names:
            # Let's provide a fix that works for the current setup
            if training_args.trainable_param_names == ".*lora_[ab].*":
                logger.warning("Overwriting `trainable_param_names` to include the KE")
                training_args.trainable_param_names = f".*.{ke_name}.lora_[ab].*"
            else:
                raise ValueError(
                    "Please ensure that the Knowledge Extractor will be trained"
                )

        # Make sure that when creating the datamodule, we only load tasks
        # for which we have trained KM experts
        if training_args.finetune_task_name:
            logger.warning("Overwriting `finetune_task_name` to match the KM experts")

        training_args.finetune_task_name = list(
            filter(lambda x: x != ke_name, model.experts_names)
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
        model.push_to_hub(args.ke_hf_path)
    elif args.library_id:
        expert_library = ExpertLibrary.get_expert_library(args.library_id)
        ke_expert = model.get_expert_instance(ke_name)
        expert_library.add_expert(ke_expert, ke_name)


if __name__ == "__main__":
    args = KEArguments.parse()
    assert args.dataset_config
    train_ke(args)
