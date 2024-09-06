from typing import Type

from pytorch_lightning import seed_everything

from mttl.arguments import ExpertConfig
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import (
    BaseExpertModel,
    BaseExpertModelConfig,
    ExpertModel,
    ExpertModelConfig,
)
from mttl.models.hf.callbacks import DownstreamEvalCallback
from mttl.models.hf.trainer import ExpertModelTrainer
from mttl.models.library.expert_library import ExpertLibrary
from mttl.utils import rank_zero_only_and_wait, remote_login


def train_experts(
    model_config: BaseExpertModelConfig,
    model_class: Type[BaseExpertModel],
    training_args: ExpertConfig,
):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)
    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    expert_library = None
    if training_args.library_id:

        @rank_zero_only_and_wait(before=False, after=True)
        def create_library(args):
            expert_library = ExpertLibrary.get_expert_library(
                repo_id=args.library_id,
                destination_id=args.destination_library_id,
                create=True,
            )
            return expert_library

        expert_library = create_library(training_args)

    module = model_class(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    )

    callbacks = []
    if training_args.pipeline_eval_tasks:
        if training_args.pipeline_eval_tasks == "all":
            training_args.pipeline_eval_tasks = "arc-challenge,arc-easy,boolq,hellaswag,humaneval,mbpp,openbookqa,piqa,bbh-fast,winogrande"

        downstream_eval_callback = DownstreamEvalCallback(module, training_args)
        callbacks.append(downstream_eval_callback)
    else:
        logger.warning(
            "Deactivating downstream eval callback as it is not enabled in the config. Please set `pipeline_eval_tasks`."
        )

    trainer: ExpertModelTrainer = ExpertModelTrainer(
        model=module,
        args=training_args,
        callbacks=callbacks,
    )
    if training_args.eval_before_training:
        trainer.evaluate()

    trainer.train()

    # Get the best checkpoint
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        logger.info("Best model checkpoint: %s", best_model_path)

    # final test
    trainer.predict(trainer.dm.test_dataset)

    # upload to library!
    if expert_library is not None:
        expert_library.add_expert_from_ckpt(best_model_path)


if __name__ == "__main__":
    args = ExpertConfig.parse()

    model_config = ExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name or args.finetune_task_name,
        modifier_config=args.modifier_config,
    )

    train_experts(model_config, ExpertModel, args)
