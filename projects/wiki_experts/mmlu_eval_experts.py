import os
import sys
import json
import torch
import wandb 
import logging
import pytorch_lightning as pl
from huggingface_hub import login
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from mttl.evaluators import MMLUEvaluator
from mttl.datamodule.wiki_mmlu_module import WikiMMLUDataModule
from mttl.utils import get_mlf_logger, setup_logging, logger

# register models
from projects.wiki_experts.expert_model import MultiExpertModel
from config import ExpertConfig


def run_eval(args):
    seed_everything(args.seed, workers=True)

    # get directory of the current file
    setup_logging(args.output_dir)

    logger.info("Args: {}".format(args.to_json()))

    if args.hf_token_hub:
        login(token=args.hf_token_hub)

    # select dataloader
    args.finetune_task_name = "college_biology,high_school_government_and_politics,prehistory,security_studies"
    mmlu = MMLUEvaluator(
        args,
        data_dir=os.environ["MMLU_DATA_DIR"],
    )
    module = MultiExpertModel(
        **vars(args),
        tokenizer=mmlu.datamodule.tokenizer
    )

    if args.experts_to_load:
        for expert in args.experts_to_load.split(","):
            expert_path, action = expert.split(":")
            module.load_expert(expert_path, action=action)
    elif args.experts_to_load == "all":
        module.load_expert("gptneo_125m_experts/infollow", action="merge")
        module.load_expert("gptneo_125m_experts/college_biology", action="route")
        module.load_expert("gptneo_125m_experts/high_school_government_and_politics", action="route")
        module.load_expert("gptneo_125m_experts/prehistory", action="route")
        module.load_expert("gptneo_125m_experts/security_studies", action="route")

    module.to("cuda")
    scores = mmlu.evaluate(module, subsample=10)
    print("MMLU Accuracy:", scores["all"]["mean"])

    del module, mmlu


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
