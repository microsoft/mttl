# Requires:
#
# pip install kvpress

import math
import json
import logging
from pathlib import Path
from typing import Optional
import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
import os
import torch
from transformers import pipeline

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.dist_utils import (
    is_main_process,
    get_device,
    get_world_size,
    get_data_sampler,
    get_local_rank,
    distributed_mean,
)
import projects.kms.utils.press_pipeline

import numpy as np
from kvpress import (
    AdaKVPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
)


logger = logging.getLogger(__name__)


def evaluate(
    dataset: str,
    data_dir: Optional[str] = None,
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device: Optional[str] = None,
    compression_ratio: float = 0.1,
    fraction: float = 1.0,
    max_new_tokens: Optional[int] = None,
    max_context_length: Optional[int] = None,
    compress_questions: bool = False,
    split: str = "test",
    output_dir: str = "test",
):
    data_dir = str(data_dir) if data_dir else None

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    np.random.seed(42)

    # Load dataframe
    dataset_name = dataset
    dataset = load_dataset(dataset, split="train")
    dataset = dataset.filter(lambda x: x["split"] == split)

    if fraction < 1.0:
        indices = np.random.choice(
            len(dataset), size=int(fraction * len(dataset)), replace=False
        )
        dataset = dataset.select(indices)
        print(indices)

    # Split in world size
    num_procs = get_world_size()
    total_len = len(dataset)
    shard_size = math.ceil(total_len / num_procs)

    start = shard_size * get_local_rank()
    end = min(start + shard_size, total_len)

    dataset = dataset.select(range(start, end))

    # Initialize pipeline with the correct attention implementation
    model_kwargs = {"torch_dtype": torch.bfloat16}
    model_kwargs["attn_implementation"] = "flash_attention_2"

    press = KnormPress()
    press.compression_ratio = compression_ratio

    pipe = pipeline(
        "l2-text-generation",
        model=model,
        device=get_device(),
        model_kwargs=model_kwargs,
    )

    if "narrativeqa" in dataset_name:
        prompt = "Answer the following question. Give only the answer, and no extra commentary, formatting, or chattiness. Question: "
    else:
        raise ValueError("")

    all_gt_answers = []
    all_answers = []
    all_rougeL = []

    pbar = tqdm(dataset, disable=not is_main_process())
    for i, example in enumerate(pbar):
        context = example["text"]
        context = f"Consider the following paragraph:\n{context}\n{prompt}"
        questions = example["questions"]
        gt_answers = example["answers"]

        output = pipe(
            context,
            questions=questions,
            max_new_tokens=64,
            max_context_length=128_000,
            press=press,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
        )

        eval_metrics = compute_metrics(output["answers"], gt_answers, reduction="none")
        all_rougeL.extend(eval_metrics["rougeL"])
        pbar.set_description(str(np.mean(all_rougeL)))
        torch.cuda.empty_cache()

    lens = distributed_mean(pipe.cmp_lengths, get_device())
    orig_lens = distributed_mean(pipe.ctx_lengths, get_device())
    rouge = distributed_mean(all_rougeL, get_device())

    if is_main_process():
        pbar.set_description(f"ROUGE: {rouge}")
        print("Final", rouge)
        output_dir = output_dir + f"/c{compression_ratio}_knorm"
        os.makedirs(output_dir, exist_ok=True)
        with open(output_dir + "/metrics.json", "w") as f:
            import json

            json.dump({"rouge_L": rouge, "cmp_lengths": lens, "ctx_lens": orig_lens}, f)


if __name__ == "__main__":
    Fire(evaluate)
