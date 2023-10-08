import torch
import sys
import tqdm
import numpy as np
import os
import gc

from datasets import load_dataset
from vllm import LLM, SamplingParams
from mmlu_subject_configs import SUB_10, SUB_10_LAST
import click
import random

sys.path.append("../../")
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import openai
from mttl.models.adapters import LoRA
from mttl.utils import setup_logging
from mttl.dataloader.platypus_dataset_reader import InversePlatypusTemplate

import time

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv(
    "AZURE_OPENAI_ENDPOINT"
)  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = "azure"
openai.api_version = "2023-05-15"  # this may change in the future

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 200,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


class OpenAI:
    def __init__(self, model_name="text-davinci-003", n=2):
        self.n_instructions_per_context = n
        self.model_name = model_name
        self.sys_prompt = f"You are a helpful and precise assistant for generating instructions for given user responses.\
                        \nYou are given some examples of valid instructions in the format ### Instruction example: <instruction>.\
                        \nThen you are given a context that contains reponces for some instruction in the format ### Context: <response>.\
                        \nYour task is to provide {self.n_instructions_per_context} new instructions to which the answer is contained in the context. Your instruction should be similar to the format of the example instructions.\
                        \nFor each of the {self.n_instructions_per_context} instructions you provide, also generate a complete reponse that is contained in the context. Format your output as follows:  ### Instruction <i>: <your instruction> ### Response <i>: <response from the context>.\
                        \nRemember, it is very important to keep your generated instruction as similar as possible in the format, style and length to the provided examples!!!"

    @retry_with_exponential_backoff
    def api_generation(
        self,
        messages: str,
        *args,
        **kwargs,
    ):
        responses = [
            openai.ChatCompletion.create(
                messages=messages,
                deployment_id=self.model_name,
                *args,
                **kwargs,
            )
        ]
        time.sleep(3)  # Preventing rate limits
        return responses

    def generate(self, templated_contexts, sampling_params, **kwargs):
        results = []       
        for context in tqdm.tqdm(templated_contexts):
            context = context.replace("\n\n### Response:", "\n\n### Context:")
            context = context.replace("\n\n### Instruction:", "\n\n### Yoru response:")
            message = [
                {"role": "system", "content": self.sys_prompt},
                {
                    "role": "user",
                    "content": context,
                },
            ]
            response = self.api_generation(message)
            results.append(response[0].choices[0]["message"]["content"])


def sample_icl_examples(dataset, n_icl):
    dataset = dataset.shuffle()
    return dataset[:n_icl]["input"]


def generate_instructions_(
    llm,
    subject_names=SUB_10,
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    num_contexts_per_document=4,
    max_context_length=512,
    output_filename="generated.jsonl",
    in_context_source="lukaemon/mmlu",
    n_icl=2,
):
    """
    To generate instructions, we take num_contexts_per_document chunks of length max_context_length from each document,
    then sample 1 instruction from each chunk.

    All instructions are appended as jsonl into output_filename.
    """
    random_sample = True
    template = InversePlatypusTemplate()
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    dataset = load_dataset("sordonia/wiki_mmlu_from_valid_all")["train"].to_pandas()

    for subject in subject_names:
        if n_icl > 0:
            icl_dataset = load_dataset(in_context_source, subject)["validation"]

        subject_data = dataset[dataset["subject"] == subject]

        contexts = []
        for text in tqdm.tqdm(subject_data["text"]):
            sentences = text.split(".")
            sentences = [
                sentence.strip().replace("\n", " ").replace("  ", " ")
                for sentence in sentences
                if len(sentence.strip()) > 0
            ]

            if not random_sample:
                append = False
                for sentence in sentences:
                    sentence = sentence + "."
                    if not append:
                        contexts.append(sentence)
                        append = True
                    else:
                        if (
                            len(contexts[-1].split()) + len(sentence.split())
                            < max_context_length
                        ):
                            contexts[-1] += " " + sentence
                        else:
                            contexts.append(sentence)
            else:
                for _ in range(num_contexts_per_document):
                    sent = ""
                    start = np.random.randint(0, len(sentences))

                    while len(sent.split()) < max_context_length:
                        sent += sentences[start] + ". "
                        start = (start + 1) % len(sentences)

                    # avoid context too long errors
                    if len(sent.split()) > max_context_length:
                        sent = " ".join(sent.split()[:max_context_length])

                    contexts.append(sent.strip())

        templated_contexts = [
            template.apply(
                {
                    "instruction": None,
                    "input": None,
                    "icl_examples": sample_icl_examples(icl_dataset, n_icl)
                    if n_icl > 0
                    else None,
                    "output": sentence,
                }
            )       
            for sentence in contexts
        ]

        outputs = llm.generate(templated_contexts, sampling_params, use_tqdm=True)
        with open(output_filename, "a+") as f:
            import json

            for output, context in zip(outputs, contexts):
                f.write(
                    json.dumps(
                        {
                            "instruction": output.outputs[0].text,
                            "context": context,
                            "subject": subject,
                        }
                    )
                )
                f.write("\n")


def generate_answers_(
    llm,
    instruction_json,
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
):
    import json

    output_filename = instruction_json.replace(".json", "_answers.json")
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

    with open(instruction_json, "r") as f:
        data = [json.loads(s) for s in f.readlines()]

    requests = []
    for instance in tqdm.tqdm(data):
        context = instance["context"]
        instruction = instance["instruction"]
        requests.append(
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Context:\n{context}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        )

    outputs = llm.generate(requests, sampling_params)
    with open(output_filename, "a+") as f:
        import json

        for output, instance in zip(outputs, data):
            instance["input"] = ""
            instance["response"] = output.outputs[0].text
            f.write(json.dumps(instance))
            f.write("\n")


def load_vllm_model(path, dtype="float16", tensor_parallel_size=2):
    llm = LLM(
        model=path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.5,
    )
    return llm


def save_merged_model(mttl_ckpt_path, hf_path="/tmp/merged"):
    from expert_trainer import ExpertTrainer
    from mttl.utils import logger

    hf_path = os.path.join(hf_path, mttl_ckpt_path.replace("/", "_"))
    if not os.path.exists(hf_path):
        os.makedirs(hf_path)
    else:
        return hf_path

    model = ExpertTrainer.from_pretrained(
        mttl_ckpt_path,
        load_in_8bit=False,
        device_map={"": "cpu"},
    )
    if not model.hparams.model_modifier == "lora":
        raise NotImplementedError("Only LoRA models are supported.")

    merged = []
    for name, module in model.model.named_modules():
        for c_name, child in module.named_children():
            if isinstance(child, LoRA):
                child.merge_with_layer()
                setattr(
                    module,
                    c_name,
                    child.layer,
                )
                merged.append(name)

    logger.info("Merged LoRA layers: %s" % merged)
    logger.info("Saving merged model to: %s" % hf_path)

    model.model.save_pretrained(hf_path, save_full_model=True)
    logger.info("Saving tokenizer to: %s" % hf_path)
    model.tokenizer.save_pretrained(hf_path)

    # free all
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return hf_path


@click.group()
def cli():
    setup_logging()


@cli.command("gena")
@click.option("--model-path", type=str, required=True)
@click.option("--instruction-json", type=str, required=True)
def generate_answers(model_path, instruction_json):
    save_merged_model(model_path, "/tmp/merged")
    llm = load_vllm_model("/tmp/merged")
    generate_answers_(llm, instruction_json=instruction_json)


@cli.command("merge-n-save")
@click.option("--mttl-checkpoint", type=str, required=True)
@click.option("--output-path", type=str, required=True)
def generate_instructions(mttl_checkpoint, output_path):
    save_merged_model(mttl_checkpoint, output_path)


@cli.command("geni")
@click.option("--model-path", type=str, required=True)
@click.option("--output-filename", type=str, required=True)
def generate_instructions(model_path, output_filename):
    if model_path in ["gpt-35-turbo", "gpt-4"]:
        llm = OpenAI(model_path)
    else:
        model_path = save_merged_model(
            model_path, "/home/v-oostapenko/mttl_out/models/merged"
        )
        llm = load_vllm_model(model_path, tensor_parallel_size=1)
    generate_instructions_(llm, output_filename=output_filename)


if __name__ == "__main__":
    cli()
