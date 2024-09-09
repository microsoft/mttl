import json
import os

import click
import torch
import tqdm
import vllm
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def chunk_wikipedia(text, tokenizer, block_size):
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
@click.option("--model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
@click.option("--block_size", type=int, default=2048)
@click.option("--max_continuation_length", type=int, default=768)
@click.option("--num_summaries", type=int, default=16)
@click.option("--generation_top_p", type=float, default=0.8)
@click.option("--top_docs_per_subject", type=int, default=50)
@click.option("--output_path", type=str, default="wiki_summaries")
def main(
    model,
    block_size,
    max_continuation_length,
    num_summaries,
    generation_top_p,
    top_docs_per_subject,
    output_path,
):
    args = {
        "model": model,
        "block_size": block_size,
        "max_continuation_length": max_continuation_length,
        "num_summaries": num_summaries,
        "generation_top_p": generation_top_p,
        "top_docs_per_subject": top_docs_per_subject,
        "output_path": output_path,
    }

    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset = load_dataset("sordonia/my-wiki-latex_mmlu_from_valid_all", split="train")

    sampling_params = SamplingParams(
        n=num_summaries,
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

    all_sections = []
    for subject in tqdm.tqdm(
        dataset.unique("subject"), desc=f"Generating summaries for subjects"
    ):
        docs_per_subject = dataset.filter(
            lambda x: x["subject"] == subject, num_proc=16
        )
        docs_per_subject = docs_per_subject.sort("score", reverse=True)

        prompts, chunks = [], []
        for doc_idx in range(top_docs_per_subject):
            doc = docs_per_subject[doc_idx]
            chunks_iterator = chunk_wikipedia(doc["text"], tokenizer, block_size)

            for tokens, chunk in chunks_iterator:
                prompts.append(
                    f"<s><|user|>\nSummarize the following text in around {int(len(tokens) / 4)} words without omitting any important details:\n********** Text **********\n"
                    + chunk
                    + f"\n********************\nSummary:<|end|>\n<|assistant|>\n"
                )
                chunks.append(chunk)

        outputs = llm.generate(prompts, sampling_params)
        sections = []
        for output, chunk in zip(outputs, chunks):
            section = {}
            section["text"] = chunk
            section["summaries"] = []
            section["subject"] = subject
            for i, response in enumerate(output.outputs):
                section["summaries"].append(response.text)
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


if __name__ == "__main__":
    main()
