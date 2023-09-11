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
    
    def find_experts(path):
        import glob

        for path in glob.glob(expert_path + "/**/csv_metrics/", recursive=True):
            yield "/".join(path.split("/")[:-2])

    if args.experts_to_load:
        for expert in args.experts_to_load.split(","):
            expert_path, action = expert.split(":")

            for expert_path in find_experts(expert_path):
                module.load_expert(expert_path, action=action)

    module.to("cuda")
    scores = mmlu.evaluate(module, subsample=10)

    logger.info("MMLU Accuracy: {}".format(scores["all"]["mean"]))

    del module, mmlu


if __name__ == "__main__":
    args = ExpertConfig.parse()
    run_eval(args)
