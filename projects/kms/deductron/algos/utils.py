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
        [
            {
                "role": "user",
                "content": "You will be given a paragraph and general information about it, your task is to come up with a continuation of it.\n\n"
                + "This is general information about the document\n\n:"
                + response
                + "\n\nThis is the paragraph:\n\n"
                + problem
                + "\n\nNow, write a continuation to the paragraph:",
            }
        ]
        for problem, response in zip(problems, responses)
    ]
    queries_no_response = [
        [
            {
                "role": "user",
                "content": "You will be given a paragraph and general information about it, your task is to come up with a continuation of it.\n\n"
                + "This is general information about the document\n\n:"
                + "[EMPTY]"
                + "\n\nThis is the paragraph:\n\n"
                + problem
                + "\n\nNow, write a continuation to the paragraph:",
            }
        ]
        for problem in problems
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


@torch.no_grad()
def logprobs_reward(
    model, tokenizer, messages, responses, labels, temperature=DEFAULT_TEMP
):
    from projects.kms.deductron.data_utils import create_joint_tensors
    from projects.kms.deductron.utils import get_logprobs

    problems = [m[-1]["content"] for m in messages]
    queries = [
        [
            {
                "role": "user",
                "content": "Read carefully the following paragraph:\n\n"
                + problem
                + "\n\nThese are additional considerations on the paragraph above:\n\n"
                + response
                + "\n\nTake into account the previous information and write a continuation to the paragraph:",
            }
        ]
        for problem, response in zip(problems, responses)
    ]
    qr, qrm, rm = create_joint_tensors(
        tokenizer,
        queries,
        labels,
    )
    log_probs = get_logprobs(
        model,
        qr,
        qrm,
        rm,
        batch_size=2,
        temperature=temperature,
    )
    torch.cuda.empty_cache()
    return (log_probs).cpu().tolist()


def summary_task_generator(prompts) -> List[Dict[str, str]]:
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
