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

sys.path.append("../../")

from mttl.models.adapters import LoRA
from mttl.utils import setup_logging
from mttl.dataloader.platypus_dataset_reader import InversePlatypusTemplate


def generate_instructions_(
    llm,
    subject_names=SUB_10,
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    num_contexts_per_document=4,
    max_context_length=512,
    output_filename="generated.jsonl"
):
    """
    To generate instructions, we take num_contexts_per_document chunks of length max_context_length from each document,
    then sample 1 instruction from each chunk.

    All instructions are appended as jsonl into output_filename.
    """
    random_sample = True
    template = InversePlatypusTemplate()
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    dataset = load_dataset("sordonia/wiki_mmlu_from_valid_all")["train"].to_pandas()

    for subject in subject_names:
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
                        if len(contexts[-1].split()) + len(sentence.split()) < max_context_length:
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
                    json.dumps({
                        "instruction": output.outputs[0].text,
                        "context": context,
                        "subject": subject,
                    })
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
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    with open(instruction_json, "r") as f:
        data = [json.loads(s) for s in f.readlines()]

    requests = []
    for instance in data:
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
    )
    return llm


def save_merged_model(mttl_ckpt_path, hf_path="/tmp/merged"):
    from expert_trainer import ExpertTrainer
    from mttl.utils import logger

    model = ExpertTrainer.load_from_checkpoint(
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

    model.model.save_pretrained(hf_path)
    logger.info("Saving tokenizer to: %s" % hf_path)
    model.tokenizer.save_pretrained(hf_path)

    # free all
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


@click.group()
def cli():
    setup_logging()


@cli.command('gena')
@click.option("--model-path", type=str, required=True)
@click.option("--instruction-json", type=str, required=True)
def generate_answers(model_path, instruction_json):
    llm = load_vllm_model(model_path)
    generate_answers_(llm, instruction_json=instruction_json)


@cli.command('merge-n-save')
@click.option("--mttl-checkpoint", type=str, required=True)
@click.option("--output-path", type=str, required=True)
def generate_instructions(mttl_checkpoint, output_path):
    save_merged_model(mttl_checkpoint, output_path)


@cli.command('geni')
@click.option("--model-path", type=str, required=True)
@click.option("--output-filename", type=str, required=True)
def generate_instructions(model_path, output_filename):
    save_merged_model(model_path, "/tmp/merged")
    llm = load_vllm_model("/tmp/merged")
    generate_instructions_(llm, output_filename=output_filename)


if __name__ == '__main__':
    cli()
