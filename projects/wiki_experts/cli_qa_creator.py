import time
import torch
import sys
import tqdm
import numpy as np
import os
import gc
import re
import json
from dataclasses import dataclass, field
from datasets import load_dataset
from typing import List
import click
from abc import ABC, abstractmethod

from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import mmlu_subject_configs  # noqa
from mttl.dataloader.platypus_dataset_reader import (
    InversePlatypusTemplate,
    PlatypusTemplate,
)
from mttl.models.adapters import LoRA
from mttl.models.openai import GPT
from mttl.utils import setup_logging


INVALID_RESPONSE = object()


def count_repeating_sentences(text):
    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    unique_sentences = set(sentences)
    return len(sentences) - len(unique_sentences)


class InstructionsGenerator(ABC):
    @dataclass
    class Response:
        contexts: List[str] = field(default_factory=list)
        outputs: List[str] = field(default_factory=list)
        cumulative_logprobs: List[float] = field(default_factory=list)
        finish_reason: List[str] = field(default_factory=list)

    @abstractmethod
    def generate(self, templated_contexts, sampling_params, **kwargs):
        pass

    def generate_prompt_for_answer(self, instruction, context):
        return self.template.apply(instruction, context)

    def generate_reverse_prompt_for_instruction(
        self, output, context, icl_examples=None
    ):
        """
        We provide the context as output (response) if and only if a response is not
        present. Otherwise, we provide the context as context in addition to the previously generated response.
        """
        return self.inverse_template.apply(
            output if output is not None else context,
            context if output is not None else None,
            icl_examples,
        )


class InversePlatypusLLM(InstructionsGenerator, LLM):
    def __init__(self, *args, **kwargs):
        LLM.__init__(self, *args, **kwargs)
        self.inverse_template = InversePlatypusTemplate()
        self.template = PlatypusTemplate()

    @property
    def model_name(self):
        return self.llm_engine.model_config.model

    def generate(self, templated_contexts, sampling_params, **kwargs):
        results = InstructionsGenerator.Response()

        for request_id, context in enumerate(templated_contexts):
            self.llm_engine.add_request(str(request_id), context, sampling_params)
        responses = self._run_engine(use_tqdm=True)

        responses_dict = {r.request_id: r for r in responses}
        for request_id, context in enumerate(templated_contexts):
            if (
                str(request_id) in responses_dict
                and len(responses_dict[str(request_id)].outputs[0].token_ids) > 0
            ):
                response = responses_dict[str(request_id)]
                results.outputs.append(response.outputs[0].text)
                results.cumulative_logprobs.append(
                    response.outputs[0].cumulative_logprob
                    / (len(response.outputs[0].token_ids) + 1e-10)
                )
                results.finish_reason.append(response.outputs[0].finish_reason)
            else:
                results.outputs.append(INVALID_RESPONSE)
                results.cumulative_logprobs.append(np.inf)
                results.finish_reason.append("invalid")
        return results


class OpenAI(InstructionsGenerator):
    def __init__(self, model_name="text-davinci-003"):
        self._model_name = model_name
        self.operator = GPT.create_lm(model_name=self.model_name)

    @property
    def model_name(self):
        return self._model_name

    def generate(self, templated_contexts, sampling_params, **kwargs):
        outputs = []
        results = InstructionsGenerator.Response()

        pbar = tqdm.tqdm(range(len(templated_contexts) // 20))
        for context in range(0, len(templated_contexts), 20):
            batch = templated_contexts[context : context + 20]
            outputs += self.operator.generate(
                batch, max_tokens=sampling_params.max_tokens
            )
            pbar.update(len(batch))
        return results


def sample_icl_examples(dataset, n_icl, use_options=True):
    dataset = dataset.shuffle()
    examples = []
    for i in range(n_icl):
        example = dataset[i]["input"]
        if use_options:
            for ans_option in ["A", "B", "C", "D"]:
                option = f"\n{ans_option}: " + dataset[i][ans_option]
                example += option
        examples.append(example)
    return examples


def reject_output(output, finish_reason):
    # common error here is that instruxtions starts with BEGININPUT + some nonsense, let evoid it
    # usually, if the instruction is too long, its a bad one
    # or if it stopped due to max tokens
    return (
        output is INVALID_RESPONSE
        or "BEGININPUT" in output
        or not output.strip()
        or finish_reason != "stop"
    )


def free_memory():
    from ray import shutdown

    gc.collect()
    torch.cuda.empty_cache()
    destroy_model_parallel()
    shutdown()
    time.sleep(3)


def transform_seed_dataset(
    dataset_name="sordonia/my-wiki-latex_mmlu_from_valid_all",
    subjects="SUB_10",
    icl_dataset_name="lukaemon/mmlu",
    icl_use_out_options=True,
    icl_examples=0,
    max_context_length=512,
    max_documents_per_subject=1e6,
    subset=1,
):
    """
    Convert a seed dataset into a tuple of (context, subject, icl_examples).
    """

    dataset = load_dataset(dataset_name)["train"].to_pandas()
    converted_dataset = []

    if type(subjects) == str:
        subjects = getattr(mmlu_subject_configs, subjects)

    for subject in subjects:
        subject_data = dataset[dataset["subject"] == subject]
        subject_data.sort_values(by="dfq", ascending=False, inplace=True)

        if icl_examples > 0:
            icl_dataset = load_dataset(icl_dataset_name, subject)["validation"]

        subject_contexts = []
        num_contexts_per_doc = [0]

        for i in tqdm.tqdm(range(len(subject_data)), desc=f"Processing {subject}..."):
            document = subject_data.iloc[i]
            text = document["text"]

            sentences = text.split(".")
            sentences = [
                sentence.strip().replace("\n", " ").replace("  ", " ")
                for sentence in sentences
                if len(sentence.strip()) > 0
            ]

            # new document
            document_contexts = []
            for sentence in sentences:
                sentence = sentence + "."
                if not document_contexts:
                    document_contexts.append(sentence)
                else:
                    if (
                        len(document_contexts[-1].split()) + len(sentence.split())
                        < max_context_length
                    ):
                        document_contexts[-1] += " " + sentence
                    else:
                        document_contexts.append(sentence)

            num_contexts_per_doc.append(len(document_contexts))
            subject_contexts.extend(
                {
                    "text": context,
                    "docno": str(document["docno"]),
                }
                for context in document_contexts
            )

            if i > len(subject_data["text"]) * float(subset):
                print("Breaking early due to subset settings.")
                break

            if i > max_documents_per_subject:
                print("Breaking early due to max_documents_per_subject settings.")
                break

        print(
            "Contexts per document (Avg/Min/Max):",
            np.mean(num_contexts_per_doc),
            np.min(num_contexts_per_doc),
            np.max(num_contexts_per_doc),
        )

        for context in subject_contexts:
            converted_dataset.append(
                {
                    "id": str(len(converted_dataset)),
                    "context": context["text"],
                    "docno": str(context["docno"]),
                    "subject": subject,
                    "icl_examples": sample_icl_examples(
                        icl_dataset, icl_examples, use_options=icl_use_out_options
                    )
                    if icl_examples > 0
                    else None,
                }
            )
    return converted_dataset


def dump_jsonl_dataset(dataset, filename):
    """Writes a jsonl dataset into a file."""
    with open(filename, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry))
            f.write("\n")


def read_jsonl_dataset(filename):
    """Read a jsonl dataset into a list of dicts."""
    with open(filename, "r") as f:
        dataset = [json.loads(s) for s in f.readlines()]
    return dataset


def generate_instructions_(
    llm: InstructionsGenerator,
    dataset,
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
):
    """
    To generate instructions, we take num_contexts_per_document chunks of length max_context_length from each document,
    then sample 1 instruction from each chunk.

    All instructions are appended as jsonl into output_filename.
    """
    import copy

    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

    def get_templated_context(entry):
        return (
            llm.generate_reverse_prompt_for_instruction(
                context=entry["context"],
                output=entry["response"] if "response" in entry else None,
                icl_examples=entry["icl_examples"],
            )
        )

    templated_contexts = [get_templated_context(entry) for entry in dataset]

    print("Example generation requests...")
    for context in np.random.choice(templated_contexts, 5):
        print(context)
        print()

    result = llm.generate(templated_contexts, sampling_params)
    assert len(result.outputs) == len(templated_contexts)

    new_dataset = []
    for (entry, instruction, finish_reason) in zip(dataset, result.outputs, result.finish_reason):
        if reject_output(instruction, finish_reason):
            continue

        copied_entry = copy.deepcopy(entry)
        copied_entry.update(
            {
                "instruction": instruction,
                "author_instr": str(llm.model_name),
            }
        )
        new_dataset.append(copied_entry)

    print("Created a new instruction dataset of size:", len(new_dataset))
    return new_dataset


def generate_answers_(
    llm: InstructionsGenerator,
    dataset,
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
):
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )

    requests = []
    for instance in tqdm.tqdm(dataset):
        requests.append(
            llm.generate_prompt_for_answer(
                instance["instruction"],
                instance["context"],
            )
        )

    result = llm.generate(requests, sampling_params)
    assert len(result.outputs) == len(requests)

    new_dataset = []
    for entry, response, log_p, reason in zip(
        dataset, result.outputs, result.cumulative_logprobs, result.finish_reason
    ):
        import copy

        if reject_output(response, reason):
            continue

        # lets also avoid outputs that contains repetitions of the same sentence more than once
        n_rep = count_repeating_sentences(response)
        if n_rep > 0:
            continue

        entry = copy.deepcopy(entry)
        entry.update(
            {
                "response": response,
                "author_response": str(llm.model_name),
                "normalized_cumul_logprob_response": log_p,
            }
        )
        new_dataset.append(entry)

    print("Created a new answer dataset of size:", len(new_dataset))
    return new_dataset


def load_vllm_model(path, dtype="float16"):
    free_memory()
    llm = InversePlatypusLLM(
        model=path,
        dtype=dtype,
        tensor_parallel_size=int(os.environ.get("VLLM_TPSIZE", 1)),
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM", 0.5)),
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
    return hf_path


@click.group()
def cli():
    setup_logging()


@cli.command("gena")
@click.option("--model-path", type=str, required=True)
@click.option("--instruction-json", type=str, required=True)
@click.option("--tmp_path", type=str, required=False, default="/tmp/merged")
def generate_answers(model_path, instruction_json, tmp_path="/tmp/merged"):
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:
        instruction_json = os.path.join(
            os.environ.get("AMLT_OUTPUT_DIR"), instruction_json
        )

    model_path = save_merged_model(model_path, hf_path=tmp_path)
    llm = load_vllm_model(model_path)
    dataset = generate_answers_(llm, dataset=read_jsonl_dataset(instruction_json))
    output_filename = instruction_json.replace(".json", "_answers.json")
    dump_jsonl_dataset(dataset, output_filename)


@cli.command("merge-n-save")
@click.option("--mttl-checkpoint", type=str, required=True)
@click.option("--output-path", type=str, required=True)
def generate_instructions(mttl_checkpoint, output_path):
    save_merged_model(mttl_checkpoint, output_path)


@cli.command("geni")
@click.option(
    "--seed-dataset", type=str, default="sordonia/my-wiki-latex_mmlu_from_valid_all"
)
@click.option("--model-path", type=str, required=True)
@click.option("--output-filename", type=str, required=True)
@click.option("--n_icl", type=int, required=False, default=0)
@click.option(
    "--icl-use-out-options",
    type=bool,
    required=False,
    default=True,
    help="if True, the output options of MMLU will be included into positive prompts examples.",
)
@click.option(
    "--subset",
    type=float,
    required=False,
    default=1,
    help="if > 0, this portion of subjects' data is processed.",
)
@click.option("--tmp_path", type=str, required=False, default="/tmp/merged")
@click.option("--sub_names", type=str, required=False, default="SUB_10")
def generate_instructions(
    seed_dataset,
    model_path,
    output_filename,
    n_icl,
    icl_use_out_options,
    subset,
    tmp_path,
    sub_names,
):
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:
        output_filename = os.path.join(
            os.environ.get("AMLT_OUTPUT_DIR"), output_filename
        )

    if model_path in ["gpt-35-turbo", "gpt-4"]:
        llm = OpenAI(model_path)
    else:
        model_path = save_merged_model(model_path, hf_path=tmp_path)
        llm = load_vllm_model(model_path)

    seed_dataset = transform_seed_dataset(
        seed_dataset,
        subjects=sub_names,
        icl_examples=n_icl,
        icl_use_out_options=icl_use_out_options,
        subset=subset,
    )
    instruction_dataset = generate_instructions_(
        llm,
        seed_dataset,
    )
    dump_jsonl_dataset(instruction_dataset, output_filename)


@cli.command("e2e")
@click.option(
    "--seed-dataset", type=str, default="sordonia/my-wiki-latex_mmlu_from_valid_all"
)
@click.option("--model-path", type=str, required=True)
@click.option("--inverse-model-path", type=str, required=True)
@click.option("--output-filename", type=str, required=True)
@click.option("--n_icl", type=int, required=False, default=0)
@click.option(
    "--icl-use-out-options",
    type=bool,
    required=False,
    default=True,
    help="if True, the output options of MMLU will be included into positive prompts examples.",
)
@click.option(
    "--subset",
    type=float,
    required=False,
    default=1,
    help="if > 0, this portion of subjects' data is processed.",
)
@click.option("--tmp_path", type=str, required=False, default="/tmp/merged")
@click.option("--sub_names", type=str, required=False, default="SUB_10")
@click.option("--num_iterations", type=int, required=False, default=1)
@click.option("--max_documents_per_subject", type=int, required=False, default=1e6)
def e2e(
    seed_dataset,
    model_path,
    inverse_model_path,
    output_filename,
    n_icl,
    icl_use_out_options,
    subset,
    tmp_path,
    sub_names,
    num_iterations,
    max_documents_per_subject,
):
    if os.environ.get("AMLT_OUTPUT_DIR") is not None:
        output_filename = os.path.join(
            os.environ.get("AMLT_OUTPUT_DIR"), output_filename
        )

    # start dataset
    prev_dataset = transform_seed_dataset(
        seed_dataset,
        subjects=sub_names,
        icl_examples=n_icl,
        icl_use_out_options=icl_use_out_options,
        subset=subset,
        max_documents_per_subject=max_documents_per_subject,
    )

    llm = None
    for i in range(num_iterations):
        if model_path in ["gpt-35-turbo", "gpt-4"] and i == 0:
            llm = OpenAI(model_path)
        else:
            if i == 0:
                inverse_model_path = save_merged_model(
                    inverse_model_path, hf_path=tmp_path
                )
            del llm
            llm = load_vllm_model(inverse_model_path)

        inst_filename = output_filename.replace(".jsonl", "_inst_%d.jsonl" % i)
        answ_filename = output_filename.replace(".jsonl", "_%d.jsonl" % i)
        if not os.path.exists(inst_filename):
            instruction_dataset = generate_instructions_(
                llm,
                prev_dataset,
            )
            dump_jsonl_dataset(instruction_dataset, inst_filename)
        else:
            instruction_dataset = read_jsonl_dataset(inst_filename)

        if model_path not in ["gpt-35-turbo", "gpt-4"]:
            if i == 0:
                model_path = save_merged_model(model_path, hf_path=tmp_path)
            del llm
            llm = load_vllm_model(model_path)

        if not os.path.exists(answ_filename):
            answer_dataset = generate_answers_(
                llm,
                instruction_dataset,
            )
            dump_jsonl_dataset(answer_dataset, answ_filename)
        else:
            answer_dataset = read_jsonl_dataset(answ_filename)
        prev_dataset = answer_dataset


if __name__ == "__main__":
    cli()
