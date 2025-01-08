import argparse
import contextlib
import gc
import itertools
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os.path import join as pjoin
from typing import Dict, List, Union

import numpy as np
import ray
import tiktoken
import torch
from accelerate import Accelerator
from datasets import load_dataset
from openai import OpenAI
from termcolor import colored
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from projects.kms.deductron.utils import DEFAULT_TEMP


def get_task(task_name: str):
    if task_name == "summary_autoencoder":
        task = SummaryAutoencoderTask()
    else:
        raise ValueError("Task not known!")
    return task


class SummaryAutoencoderTask:
    def get_rewards(
        self, model, tokenizer, messages, responses, labels, temperature=DEFAULT_TEMP
    ):
        from projects.kms.deductron.data_utils import create_joint_tensors
        from projects.kms.deductron.utils import get_logprobs
        from projects.kms.deductron.ddp_utils import ddp_state

        problems = [m[-1]["content"] for m in messages]
        queries = [
            self.decode_template(problem, response)
            for problem, response in zip(problems, responses)
        ]
        # for auto-encoding, we are trying to reconstruct the paragraph itself!
        qr, qrm, rm = create_joint_tensors(
            tokenizer, queries, problems, max_length=4096
        )
        log_probs = get_logprobs(
            model,
            qr,
            qrm,
            rm,
            batch_size=4,
            temperature=temperature,
        )
        del qr, qrm, rm
        torch.cuda.empty_cache()
        queries_empty = [
            self.decode_template(problem, "[EMPTY]") for problem in problems
        ]
        # for auto-encoding, we are trying to reconstruct the paragraph itself!
        qr, qrm, rm = create_joint_tensors(
            tokenizer, queries_empty, problems, max_length=4096
        )
        log_probs_base = get_logprobs(
            model,
            qr,
            qrm,
            rm,
            batch_size=4,
            temperature=temperature,
        )
        torch.cuda.empty_cache()
        return (log_probs - log_probs_base).cpu().tolist()

    def decode_template(self, problem: str, summary: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "user",
                "content": "You are provided with a summary created from a hidden paragraph of text, and the first sentence of the paragraph to give you a rough idea."
                + " Your task is to write the paragraph from the information contained in the summary.\n"
                + "This is the summary of the paragraph\n\n:"
                + summary
                + "\n\nThese are the first few sentences of the paragraph:"
                + "\n".join(problem.split("\n")[:10])
                + "\n\nNow, utilize your best judgment and try to recreate the full paragraph:",
            }
        ]

    def encode_template(self, prompts) -> List[Dict[str, str]]:
        return [
            [
                {
                    "role": "user",
                    "content": "Read carefully the following text in order to gather all the important information.\n\n"
                    + prompt
                    + f"\n\nPlease, provide a summary of the text of no more than {len(prompt) // 4} words:",
                }
            ]
            for prompt in prompts
        ]


def next_chunk_prediction_task(problem, response):
    return [
        {
            "role": "user",
            "content": "You will be given a paragraph and general information about it, your task is to come up with a continuation of it.\n\n"
            + "This is general information about the paragraph\n\n:"
            + response
            + "\n\nThis is the paragraph:\n\n"
            + problem
            + "\n\nNow, utilize your best judgment and write a continuation to the paragraph:",
        }
    ]


@torch.no_grad()
def infogain_reward(
    model, tokenizer, messages, responses, labels, temperature=DEFAULT_TEMP
):
    from projects.kms.deductron.data_utils import create_joint_tensors
    from projects.kms.deductron.utils import get_logprobs
    from projects.kms.deductron.ddp_utils import ddp_state

    old_device = model.device
    model.to(ddp_state.device)

    problems = [m[-1]["content"] for m in messages]
    queries = [
        next_chunk_prediction_task(problem, response)
        for problem, response in zip(problems, responses)
    ]
    queries_no_response = [
        next_chunk_prediction_task(problem, "[EMPTY]") for problem in problems
    ]
    qr, qrm, rm = create_joint_tensors(
        tokenizer,
        queries,
        labels,
    )
    qr_no, qrm_no, rm_no = create_joint_tensors(
        tokenizer,
        queries_no_response,
        labels,
    )
    log_probs = get_logprobs(
        model,
        qr,
        qrm,
        rm,
        batch_size=4,
        temperature=temperature,
    )
    log_probs_no = get_logprobs(
        model,
        qr_no,
        qrm_no,
        rm_no,
        batch_size=4,
        temperature=temperature,
    )
    del qr, qrm, rm, qr_no, qrm_no, rm_no
    torch.cuda.empty_cache()
    model.to(old_device)
    return (log_probs - log_probs_no).cpu().tolist()
