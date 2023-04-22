# global interpreter
from typing import List
import os
import warnings
import asyncio
import openai
import torch
from torch import nn
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


forward_interpreter = None
backward_interpreter = None
debug_mode = False


class GPT3:
    AVAILABLE_MODELS = [
        "gpt-3.5-turbo",
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "gpt-4",
        # "gpt-4-32k",  # Not available for now
    ]

    def __init__(self, model_name="gpt-3.5-turbo", **generation_options):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"model_name should be one of: {','.join(self.AVAILABLE_MODELS)}"
            )

        self.generation_options = generation_options
        self.engine = model_name

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
        ),
    )
    async def aget_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        response = await openai.ChatCompletion.acreate(
            model=self.engine,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        output = response["choices"][0]["message"]["content"].strip()
        return output

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
        ),
    )
    def get_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        output = response["choices"][0]["message"]["content"].strip()
        return output

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
        ),
    )
    def get_completion_response(self, prompt_batch, return_logprobs=False, **kwargs):
        """
        prompting gpt-3 via openai api
        now batching only works for completion, not on chat
        """
        logging.info(kwargs)
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt_batch,
            logprobs=1,
            **kwargs,
        )

        output = []
        nlls = []
        lengths = []

        for response in response["choices"]:
            output.append(response["text"].strip())
            nlls.append(sum(response["logprobs"]["token_logprobs"]))
            lengths.append(len(response["logprobs"]["token_logprobs"]))

        if return_logprobs:
            output = list(zip(output, nlls, lengths))
        return output

    async def gather_chat_response(self, inputs, **generation_options):
        outputs = await asyncio.gather(*[self.aget_chat_completion_response(_input, **generation_options) for _input in inputs])
        return outputs

    def generate(self, inputs, async_generation=True, **kwargs):
        if type(inputs) is not list:
            inputs = [inputs]

        if debug_mode:
            print(
                "Called generation for:",
                inputs,
            )

        kwargs.pop("output_space", None)
        generation_options = self.generation_options.copy()
        generation_options.update(**kwargs)

        print("generation_options: ", generation_options)

        if self.engine in ("gpt-3.5-turbo", "gpt-4", "gpt-4-32k"):
            if "return_logprobs" in generation_options:
                del generation_options["return_logprobs"]

            if async_generation is True:
                # async
                outputs = asyncio.run(self.gather_chat_response(inputs, **generation_options))
            else:
                # call api one by one
                outputs = [
                    self.get_chat_completion_response(_input, **generation_options)
                    for _input in inputs
                ]
        else:
            # devide to mini batches (max batch size = 20 according to openai)
            max_batch_size = 20
            input_length = len(inputs)
            num_batches = input_length // max_batch_size + (
                1 if input_length % max_batch_size > 0 else 0
            )
            outputs = []
            for i in range(num_batches):
                input_batch = inputs[max_batch_size * i : max_batch_size * (i + 1)]
                output_batch = self.get_completion_response(
                    input_batch, **generation_options
                )
                outputs = outputs + output_batch

        if debug_mode:
            print("Generated", outputs)
        return outputs


class ImmutableLM(nn.Module):
    def __init__(self, model_path, **gen_options):
        super(ImmutableLM, self).__init__()

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto"
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # set pad token id
        self.backbone.config.pad_token_id = self.backbone.config.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone_name = model_path
        self.gen_options = gen_options

    @torch.no_grad()
    def evaluate(
        self,
        inputs,
        **kwargs,
    ):
        batch = self.tokenizer(
            inputs, return_tensors="pt", padding=True
        ).input_ids.cuda()
        attention_mask = (batch != self.backbone.config.eos_token_id)
        outputs = self.backbone(
            batch,
            attention_mask=attention_mask,
        )
        lm_logits = outputs[0]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.backbone.config.eos_token_id, reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        loss = loss.view(shift_logits.size(0), shift_logits.size(1))
        loss = (loss * attention_mask[:, :-1]).sum(1).cpu()
        return (loss / attention_mask[:, :-1].sum(1).cpu()).tolist()

    @torch.no_grad()
    def generate(
        self,
        inputs,
        output_space=None,
        **kwargs,
    ):
        if output_space is not None:
            raise NotImplementedError(
                "Output space restrictions are not implemented yet."
            )

        batch = self.tokenizer(
            inputs, return_tensors="pt", padding=True
        ).input_ids.cuda()
        generated = self.backbone.generate(
            batch,
            attention_mask=(batch != self.backbone.config.eos_token_id),
            max_new_tokens=10,
            min_new_tokens=1,
            no_repeat_ngram_size=2,
            **self.gen_options,
        )
        outputs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # strip the input from the outputs
        return [
            output[len(input) :].strip("\n") for input, output in zip(inputs, outputs)
        ]


class GPT2:
    def __init__(self, model_name, **generation_options):
        # ValueError: The following `model_kwargs` are not used by the model: ['max_tokens']
        generation_options.pop("max_tokens", None)
        self.model = ImmutableLM(model_name, **generation_options).cuda()

    def evaluate(self, inputs):
        return self.model.evaluate(inputs)

    def generate(self, inputs, **kwargs):
        if type(inputs) is not list:
            inputs = [inputs]
        if debug_mode:
            print("Called generation for:", inputs)
        outputs = self.model.generate(inputs)
        if debug_mode:
            print("Generated", outputs)
        return outputs


class Random:
    sentences = [f"sentence_{i}" for i in range(100)]

    def generate(self, inputs):
        if type(inputs) is not list:
            inputs = [inputs]
        return [f"GEN[{input}]" for input in inputs]


def forward_instantiate(
    model_type="gpt2",
    model_name="distilgpt2",
    **generation_options
):
    global forward_interpreter

    if forward_interpreter is None:
        if model_type == "gpt2":
            forward_interpreter = GPT2(model_name, **generation_options)
        elif model_type == "gpt3":
            forward_interpreter = GPT3(model_name, **generation_options)
        else:
            warnings.warn(f"Invalid model type for forward: {model_type}, falling back to random.")
            forward_interpreter = Random()
    else:
        raise ValueError("Forward interpreter already instantiated.")


def backward_instantiate(
    model_type="gpt2",
    model_name="distilgpt2",
    **generation_options
):
    global backward_interpreter

    if backward_interpreter is None:
        if model_type == "gpt2":
            backward_interpreter = GPT2(model_name, **generation_options)
        elif model_type == "gpt3":
            backward_interpreter = GPT3(model_name, **generation_options)
        else:
            warnings.warn(f"Invalid model type for backward: {model_type}, falling back to random.")
            backward_interpreter = Random()
    else:
        raise ValueError("Backward interpreter already instantiated.")


def forward_evaluate(input: List[str], **kwargs):
    return forward_interpreter.generate(input, **kwargs)


def backward_evaluate(input: List[str], **kwargs):
    return backward_interpreter.generate(input, **kwargs)
