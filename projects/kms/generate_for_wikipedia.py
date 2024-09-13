import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

import click
import tenacity
import torch
import tqdm
import vllm
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from openai import AsyncAzureOpenAI
from tqdm import tqdm as ttqdm
from tqdm.asyncio import tqdm as tqdm_async
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

client = AsyncAzureOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("OPENAI_BASE_URL"),
    api_version=os.environ.get("OPENAI_API_VERSION"),
)


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


def chunk_text(text, tokenizer, block_size):
    chunks = text.split("\n")
    chunks = [tokenizer.encode(chunk, add_special_tokens=False) for chunk in chunks]
    sep = tokenizer.encode("\n", add_special_tokens=False)

    c = 0
    s = 0
    while c < len(chunks):
        curr = []
        while c < len(chunks):
            if len(curr) + len(sep) + len(chunks[c]) - s <= block_size:
                if curr:
                    curr += sep
                curr += chunks[c][s:]
                c += 1
                s = 0
            elif not curr:
                curr += chunks[c][s : s + block_size]
                s += block_size
            else:
                break

        yield curr, tokenizer.decode(curr)


@click.command()
@click.option("--model", type=str, default="microsoft/Phi-3-medium-4k-instruct")
@click.option(
    "--dataset", type=str, default="sordonia/my-wiki-latex_mmlu_from_valid_all"
)
@click.option("--block_size", type=int, default=2048)
@click.option("--max_continuation_length", type=int, default=768)
@click.option("--num_generations", type=int, default=16)
@click.option("--generation_top_p", type=float, default=0.8)
@click.option("--top_docs_per_subject", type=int, default=50)
@click.option("--use_prompts", type=str, default="summary")
@click.option("--output_path", type=str, default="wiki_summaries")
@click.option("--push_to_hub", type=str, default=None)
def main(
    model,
    dataset,
    block_size,
    max_continuation_length,
    num_generations,
    generation_top_p,
    top_docs_per_subject,
    use_prompts,
    output_path,
    push_to_hub,
):
    args = {
        "model": model,
        "dataset": dataset,
        "block_size": block_size,
        "max_continuation_length": max_continuation_length,
        "num_generations": num_generations,
        "generation_top_p": generation_top_p,
        "top_docs_per_subject": top_docs_per_subject,
        "use_prompts": use_prompts,
        "output_path": output_path,
    }

    dataset = load_dataset("sordonia/my-wiki-latex_mmlu_from_valid_all", split="train")

    if "gpt" not in model:
        tokenizer = AutoTokenizer.from_pretrained(model)
        sampling_params = SamplingParams(
            n=num_generations,
            top_p=generation_top_p,
            stop_token_ids=[tokenizer.eos_token_id],
            max_tokens=max_continuation_length,
        )
        llm = LLM(
            model=model,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        oai = False
    else:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-medium-4k-instruct")
        oai = True
        llm = None

    all_sections = []
    for subject in tqdm.tqdm(
        dataset.unique("subject"), desc=f"Generating data for subjects"
    ):
        docs_per_subject = dataset.filter(
            lambda x: x["subject"] == subject, num_proc=16
        )
        docs_per_subject = docs_per_subject.sort("score", reverse=True)

        prompts, chunks, types = [], [], []
        for doc_idx in range(top_docs_per_subject):
            doc = docs_per_subject[doc_idx]
            chunks_iterator = chunk_text(doc["text"], tokenizer, block_size)

            for tokens, chunk in chunks_iterator:
                # Summary view of the input document
                if "summary" in use_prompts:
                    prompts.append(
                        f"Summarize the following text in around {int(len(tokens) / 4)} words without omitting any important details:\n********** Text **********\n"
                        + chunk
                        + f"\n********************"
                    )
                    if not oai:
                        prompts[-1] = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompts[-1]}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    types.append("summary")
                    chunks.append(chunk)

                # QA view of the input document
                if "qa" in use_prompts:
                    prompts.append(
                        f"Create one question that can be answerable from the following text, and answer it.\n********** Text **********\n"
                        + chunk
                        + f"\n********************\n"
                    )
                    if not oai:
                        prompts[-1] = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompts[-1]}],
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    types.append("qa")
                    chunks.append(chunk)

        sections = []
        if not oai:
            outputs = llm.generate(prompts, sampling_params)
        else:
            outputs = asyncio.run(
                oai_get_completions_batched(
                    prompts,
                    model,
                    num_completions=num_generations,
                    top_p=generation_top_p,
                    max_tokens=max_continuation_length,
                )
            )

        for generation_output, chunk, type in zip(outputs, chunks, types):
            section = {}
            section["input"] = chunk
            section["outputs"] = []
            section["subject"] = subject
            section["type"] = type
            for i, response in enumerate(generation_output.outputs):
                section["outputs"].append(response.text)
            sections.append(section)
        all_sections.extend(sections)

        os.makedirs(output_path, exist_ok=True)

        d = DatasetDict(
            {
                "train": Dataset.from_list(
                    all_sections, info=DatasetInfo(description=json.dumps(args))
                )
            }
        )
        d.save_to_disk(output_path)

        if push_to_hub is not None:
            d.push_to_hub(push_to_hub)


if __name__ == "__main__":
    main()
