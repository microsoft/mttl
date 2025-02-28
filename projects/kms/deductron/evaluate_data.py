import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

import asyncio
from dataclasses import dataclass
from functools import partial, wraps
from hashlib import md5
import json
import time
import numpy as np
from typing import Optional, List, Any, Callable

from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError
import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import tiktoken
import os
import datasets
import re
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
from mttl.dataloader.ni_metrics import compute_metrics


np.random.seed(10)
api_version = '2024-10-21'
model_name = 'gpt-4o-mini'
model_version = '2024-07-18'
scope = "api://trapi/.default"
credential = get_bearer_token_provider(AzureCliCredential(), scope)
global_deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')
instance = 'msrne/shared'
endpoint = f'https://trapi.research.microsoft.com/{instance}'
global_azure_openai_async_client = None


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class JsonKVStorage():
    namespace: str = None
    working_dir: str = None

    def __post_init__(self):
        os.makedirs(self.working_dir, exist_ok=True)
        self._file_name = os.path.join(self.working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def drop(self):
        self._data = {}


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete(
    deployment_name, prompt, system_prompt=None, **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    hashing_kv = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=api_version,
        )
    return global_azure_openai_async_client


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, **kwargs
) -> str:
    return await azure_openai_complete(
        global_deployment_name,
        prompt,
        system_prompt=system_prompt,
        **kwargs,
    )


async def aquery(query: str, call_func: Callable, system_prompt: str) -> str:
    try:
        response = await call_func(
            query,
            system_prompt=system_prompt,
        )
    except:
        return "1"
    return response


def query_document(queries, call_func, system_prompt):
    loop = always_get_an_event_loop()
    return loop.run_until_complete(
        asyncio.gather(
            *[aquery(query, call_func, system_prompt) for query in queries]
        )
    )


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def load_args(args_file):
    with open(args_file, "r") as f:
        return json.load(f)


EVALUATION_PROMPT = """
You will be given some notes about a paragraph and the paragraph. Your task is to judge to what extent the notest are relevant to the paragraph,
and they are a faithful representation of the content of the paragraph. You should answer with a number between 1 and 5, where 1 means the notes are not relevant to the paragraph at all, and 5 means the notes are a perfect representation of the story in the paragraph.
Respond with ONLY the number between 1 and 5, without any explanation.

*** Paragraph ***
{paragraph}

*** Notes ***
{notes}

Score [1-5]:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="generation_output.jsonl", help="File to store outputs")
    args = parser.parse_args()
    
    output_dir = args.input_file.replace(".jsonl", "/")
    llm_response_cache = JsonKVStorage(working_dir=output_dir, namespace="llm_response_cache")
    call_func = limit_async_func_call(16)(partial(azure_gpt_4o_complete, hashing_kv=llm_response_cache))

    with open(args.input_file, "r") as f:
        lines = f.readlines()

    paragraphs = [json.loads(line)["prompt"][-1]['content'] for line in lines]
    targets = paragraphs[1:]
    notes = [json.loads(line)["response"] for line in lines][:-1]

    prompts = []
    for i, _ in enumerate(targets):
        prompts.append(
            EVALUATION_PROMPT.format(
                notes=notes[i],
                paragraph=targets[i],
            )
        )

    results = query_document(prompts, call_func, system_prompt="You are a helpful assistant. Please answer the question as best as you can.")
    with open(args.input_file.replace(".jsonl", "_evaluation.jsonl"), "w") as f:
        score = []
        for i, result in enumerate(results):
            # parse score
            score.append(int(result.strip().split('\n')[0]))
            f.write(json.dumps({"prompt": prompts[i], "response": result}) + "\n")

        import numpy

        print("Average score:", numpy.mean(score))


if __name__ == "__main__":
    main()
