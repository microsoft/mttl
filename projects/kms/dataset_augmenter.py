import asyncio
import json
import os
from dataclasses import dataclass
from typing import List, Union

import click
import tenacity
import torch
import tqdm
import vllm
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from tqdm import tqdm as ttqdm
from tqdm.asyncio import tqdm as tqdm_async
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from mttl.logging import logger
from mttl.registrable import Registrable

global client


@dataclass
class OAIGenerations:
    outputs: List = None


@dataclass
class OAIGeneration:
    text: str = None


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=10, max=60),
    stop=tenacity.stop_after_attempt(100),
)
async def oai_get_completions(
    prompt, model, num_completions=1, top_p=0.8, max_tokens=768
):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        top_p=top_p,
        max_tokens=max_tokens,
        n=num_completions,
    )
    # make the results to be in the same format as VLLM
    return OAIGenerations(
        outputs=[
            OAIGeneration(text=choice.message.content) for choice in response.choices
        ]
    )


async def oai_get_completions_batched(
    prompts, model, num_completions, top_p, max_tokens
):
    results = []
    for i in range(0, len(prompts), 100):
        batch_prompts = prompts[i : i + 100]
        batch = await tqdm_async.gather(
            *[
                oai_get_completions(p, model, num_completions, top_p, max_tokens)
                for p in batch_prompts
            ]
        )
        results.extend(batch)
    return results


def chunk_text(text, tokenizer, block_size, token_overlap=0):
    """
    A generator that chunks the given text into blocks of size `block_size` with `token_overlap` between chunks using the provided tokenizer.

    Args:
    text (str): The long document to be chunked.
    tokenizer: The tokenizer object to use for tokenizing the text.
    block_size (int): Maximum size of each chunk (in tokens).
    token_overlap (int): Number of tokens to overlap between consecutive chunks.

    Yields:
    str: A chunk of text with a maximum size of `block_size` tokens and `token_overlap` overlap.
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Create the chunks with overlap
    start = 0
    while start < len(tokens):
        end = start + block_size
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        yield chunk_tokens, chunk_text

        # Move the start position by block_size - token_overlap
        start += block_size - token_overlap


class GenerationTask(Registrable):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_prompt(self, text):
        return text

    def create_task(self, text, add_chat_template=True):
        task = self.get_prompt(text)
        if add_chat_template:
            task = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": task}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return task


@GenerationTask.register("qa")
class QATask(GenerationTask):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_prompt(self, text):
        return (
            f"Create one question that can be answerable from the following text, and answer it.\n********** Text **********\n"
            + text
            + f"\n********************\n"
        )


@GenerationTask.register("summary")
class SummaryTask(GenerationTask):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_prompt(self, text):
        return (
            f"Summarize the following text in around {int(len(text) / 4)} words without omitting any important details:\n********** Text **********\n"
            + text
            + f"\n********************"
        )


class DatasetAugmenter:
    def __init__(
        self,
        model,
        block_size,
        max_continuation_length,
        num_generations,
        generation_top_p,
    ):
        self.tasks = []
        self.model = model
        self.block_size = block_size
        self.max_continuation_length = max_continuation_length
        self.num_generations = num_generations
        self.generation_top_p = generation_top_p

        self.oai = "gpt" in model
        if not self.oai:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.sampling_params = SamplingParams(
                n=num_generations,
                top_p=generation_top_p,
                stop_token_ids=[self.tokenizer.eos_token_id],
                max_tokens=max_continuation_length,
            )
            self.llm = LLM(
                model=model,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=torch.cuda.device_count(),
                enforce_eager=True,
                max_num_seqs=64,
                max_model_len=4096,
            )
            logger.warning("VLLM: Setting max_model_len to 4096!!!")
        else:
            from openai import AsyncAzureOpenAI

            global client

            client = AsyncAzureOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                azure_endpoint=os.environ.get("OPENAI_BASE_URL"),
                api_version=os.environ.get("OPENAI_API_VERSION"),
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-medium-4k-instruct"
            )
            self.llm = None

    def add_task(self, task):
        task = GenerationTask.get_class_by_name(task)(self.tokenizer)
        self.tasks.append(task)
        return self

    def augment(
        self,
        dataset: Dataset,
    ) -> Dataset:

        prompts, chunks, types, output_dataset, rests = [], [], [], [], []

        for doc_idx in range(len(dataset)):
            text = dataset[doc_idx]["text"]
            chunks_iterator = chunk_text(text, self.tokenizer, self.block_size)

            for tokens, chunk in chunks_iterator:
                for task in self.tasks:
                    chunks.append(chunk)
                    types.append(task.registered_name)
                    prompts.append(
                        task.create_task(chunk, add_chat_template=not self.oai)
                    )
                    # append the rest of the columns
                    rests.append({})
                    for column in dataset.column_names:
                        if column != "text":
                            rests[-1][column] = dataset[doc_idx][column]

        if not self.oai:
            outputs = self.llm.generate(prompts, self.sampling_params)
        else:
            outputs = asyncio.run(
                oai_get_completions_batched(
                    prompts,
                    self.model,
                    num_completions=self.num_generations,
                    top_p=self.generation_top_p,
                    max_tokens=self.max_continuation_length,
                )
            )

        for generation_output, chunk, type, rest in zip(outputs, chunks, types, rests):
            section = {}
            section["input"] = chunk
            section["type"] = type
            section["outputs"] = []
            for i, response in enumerate(generation_output.outputs):
                section["outputs"].append(response.text)
            section.update(rest)
            output_dataset.append(section)

        d = Dataset.from_list(output_dataset)
        return d
