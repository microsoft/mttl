import copy
import json
import logging
from dataclasses import dataclass

from lightning_fabric import seed_everything

# register this datamodule!
from nqa_datamodule import NQADatamodule

from mttl.arguments import MultiExpertConfig
from mttl.logging import setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.hf.trainer import LMTrainer
from mttl.models.km_model import KMMoEModel, KMMoEModelConfig
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import remote_login

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class KEArguments(MultiExpertConfig):
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = None
    # Where to save the KE expert
    ke_hf_path: str = None


def train_ke(training_args):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    if training_args.library_id:
        logger.info("Loading expert library: %s", training_args.library_id)

        model_config = KMMoEModelConfig(
            base_model=training_args.model,
            library_id=training_args.library_id,
            expert_selection=args.finetune_task_name,
        )
        model = KMMoEModel(model_config)

        if model.ke_expert_name not in training_args.trainable_param_names:
            # Let's provide a fix that works for the current setup
            logger.warning("Overwriting `trainable_param_names` to include the KE")
            training_args.trainable_param_names = f".*{model.ke_expert_name}.*"

        # for which we have trained KM experts
        if not training_args.finetune_task_name:
            logger.info(
                f"Setting `finetune_task_name` to match with experts in the model"
            )
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

    if args.nqa_dataset is None:
        logger.info(f"Setting callback dataset to {args.dataset}")
        args.nqa_dataset = args.dataset

    # Allow to set trainable tasks from a json split file (e.g. nqa_mini_split.json)
    if isinstance(args.finetune_task_name, str) and args.finetune_task_name.endswith(
        ".json"
    ):
        with open(args.finetune_task_name, "r") as f:
            split_dict = json.load(f)

        tasks = split_dict["train"] + split_dict["dev"]
        logger.info(f"Setting finetune_task_name to {tasks}")
        args.finetune_task_name = tasks

    train_ke(args)
