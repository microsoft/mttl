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
from datasets import load_dataset
from openai import OpenAI
from termcolor import colored
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams

from projects.kms.deductron.utils import DEFAULT_TEMP


def get_task(task_name: str):
    if task_name == "s_ae":
        task = SummaryAutoencoderTask()
    elif task_name == "s_ncp":
        task = SummaryNextChunkPredictionTask()
    elif task_name == "think_ncp":
        task = ThinkNextChunkPredictionTask()
    else:
        raise ValueError("Task not known!")
    return task


class SummaryAutoencoderTask:
    def get_rewards(self, model, tokenizer, requests, temperature=DEFAULT_TEMP):
        from projects.kms.deductron.data_utils import create_joint_tensors
        from projects.kms.deductron.utils import get_logprobs

        messages = [r.messages for r in requests]
        responses = [r.response for r in requests]
        finished = [r.finished for r in requests]

        problems = [m[-1]["content"] for m in messages]
        queries = [
            self.decode_template(problem, response)
            for problem, response in zip(problems, responses)
        ]
        # for auto-encoding, we are trying to reconstruct the paragraph itself!
        qr, qrm, rm = create_joint_tensors(
            tokenizer, queries, problems, finished, max_length=4096, pad_to_length=4096
        )
        log_probs = get_logprobs(
            model,
            qr,
            qrm,
            rm,
            batch_size=2,
            temperature=temperature,
        )
        del qr, qrm, rm
        torch.cuda.empty_cache()
        queries_empty = [
            self.decode_template(problem, "[EMPTY]") for problem in problems
        ]
        # for auto-encoding, we are trying to reconstruct the paragraph itself!
        qr, qrm, rm = create_joint_tensors(
            tokenizer,
            queries_empty,
            problems,
            finished,
            max_length=4096,
            pad_to_length=4096,
        )
        log_probs_base = get_logprobs(
            model,
            qr,
            qrm,
            rm,
            batch_size=2,
            temperature=temperature,
        )
        del qr, qrm, rm
        torch.cuda.empty_cache()
        rewards = (log_probs - log_probs_base).cpu().tolist()
        return rewards

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
                    "content": f"Summarize the following text in around {int(len(prompt) / 4)} words without omitting any important details.\n"
                    + "The summary should be grammatically correct and summarize all the different sections in the text.\n"
                    + "********** Text **********\n"
                    + prompt
                    + "\n********************",
                }
            ]
            for prompt in prompts
        ]


class SummaryNextChunkPredictionTask:
    def get_rewards(self, model, tokenizer, requests, temperature=DEFAULT_TEMP):
        from projects.kms.deductron.data_utils import create_joint_tensors
        from projects.kms.deductron.utils import get_logprobs

        messages = [r.messages for r in requests]
        responses = [r.response for r in requests]
        labels = [r.label for r in requests]
        finished = [r.finished for r in requests]

        problems = [m[-1]["content"] for m in messages]
        queries = [
            self.decode_template(problem, response)
            for problem, response in zip(problems, responses)
        ]
        # for auto-encoding, we are trying to reconstruct the paragraph itself!
        qr, qrm, rm = create_joint_tensors(
            tokenizer, queries, labels, finished, max_length=4096, pad_to_length=4096
        )
        log_probs = get_logprobs(
            model,
            qr,
            qrm,
            rm,
            batch_size=2,
            temperature=temperature,
        )
        del qr, qrm, rm
        torch.cuda.empty_cache()
        queries_empty = [
            self.decode_template(problem, "[EMPTY]") for problem in problems
        ]
        # for auto-encoding, we are trying to reconstruct the paragraph itself!
        qr, qrm, rm = create_joint_tensors(
            tokenizer,
            queries_empty,
            labels,
            finished,
            max_length=4096,
            pad_to_length=4096,
        )
        log_probs_base = get_logprobs(
            model,
            qr,
            qrm,
            rm,
            batch_size=2,
            temperature=temperature,
        )
        del qr, qrm, rm
        torch.cuda.empty_cache()
        rewards = (log_probs - log_probs_base).cpu().tolist()
        return rewards

    def decode_template(self, problem, response):
        return [
            {
                "role": "user",
                "content": "You are provided with a summary created from a hidden paragraph of text, and the last few sentences of the paragraph to give you a rough idea."
                + " Your task is to write a continuation of the hidden paragraph from the information contained in the summary.\n"
                + "This is the summary of the paragraph\n\n:"
                + response
                + "\n\nThese are the last few sentences of the paragraph:"
                + "\n".join(problem.split("\n")[-10:])
                + "\n\nNow, utilize your best judgment and try to write the continuation of the hidden paragraph:",
            }
        ]

    def encode_template(self, prompts) -> List[Dict[str, str]]:
        return [
            [
                {
                    "role": "user",
                    "content": f"Summarize the following text in around {int(len(prompt) / 4)} words without omitting any important details.\n"
                    + "The summary should be grammatically correct and summarize all the different sections in the text.\n"
                    + "********** Text **********\n"
                    + prompt
                    + "\n********************",
                }
            ]
            for prompt in prompts
        ]


class ThinkNextChunkPredictionTask(SummaryNextChunkPredictionTask):
    def decode_template(self, problem, response):
        return [
            {
                "role": "user",
                "content": "You are provided with some notes created from a hidden paragraph of text, and the last few sentences of the paragraph to give you a rough idea."
                + " Your task is to write a continuation of the hidden paragraph from the information contained in the notes.\n"
                + "These are the notes of the paragraph\n\n:"
                + response
                + "\n\nThese are the last few sentences of the paragraph:"
                + "\n".join(problem.split("\n")[-10:])
                + "\n\nNow, utilize your best judgment and try to write the continuation of the hidden paragraph:",
            }
        ]

    def encode_template(self, prompts) -> List[Dict[str, str]]:
        return [
            [
                {
                    "role": "user",
                    "content": f"Think about the events and the meaning of the following text.\n"
                    + "Your thought will be used to predict other important developments in the story.\n"
                    + "********** Text **********\n"
                    + prompt
                    + "\n********************",
                }
            ]
            for prompt in prompts
        ]
