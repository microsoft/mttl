import gc
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from os.path import join as pjoin

import numpy as np
import tiktoken
import torch
from accelerate import Accelerator
from datasets import load_dataset
from openai import OpenAI
from termcolor import colored
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.dataset_augmenter import chunk_text

from mttl.models.expert_model import (
    MultiExpertModel,
    MultiExpertModelConfig,
    disable_modifiers,
    set_active_expert,
)
from mttl.models.modifiers.lora import LoRAConfig

forward_prompts = [
    "Recite important facts from the preceding text.",
    #   "Create a set of questions and answers that encompass important parts of the preceding text.",
    #   "Write a summary of the preceding text.",
]


backward_prompts = [
    "Create a paragraph given the preceding information.",
]


def get_peft_model(model_name):
    config = MultiExpertModelConfig(model_name)
    model = MultiExpertModel(config)
    model.add_empty_expert(
        "km", LoRAConfig(lora_rank=16, modify_layers=".*q.*|.*v.*|.*fc.*")
    )
    model.set_default_expert("km")
    return model.to(device)


def pad(
    tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right"
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full(
        (len(tensors), *output_shape),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=False, unbiased_variance=False):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(
        values, mask, unbiased=unbiased_variance
    )
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class VLLMEndpoint:
    """Spin up the VLLM endpoint with Docker, based on the current trained model."""

    def __init__(self, model, tokenizer, tensor_parallel_size=1):
        self.model = model
        self.tokenizer = tokenizer
        self.tensor_parallel_size = tensor_parallel_size

    def __enter__(self):
        self.start_vllm_server()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.free_memory()

    def start_vllm_server(self, host="0.0.0.0", port=8000):
        import subprocess

        if isinstance(self.model, MultiExpertModel):
            self.model.merge_and_save_base_model("/tmp/saved_model", "km", "cuda")
        else:
            self.model.save_pretrained("/tmp/saved_model")
        self.tokenizer.save_pretrained("/tmp/saved_model")

        command = [
            "docker",
            "run",
            "--runtime",
            "nvidia",
            "-v",
            "/tmp/saved_model:/tmp/saved_model",
            "-p",
            "8001:8000",
            "--ipc=host",
            "vllm/vllm-openai:latest",
            "--model",
            "/tmp/saved_model",
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
        )

        # wait for the server to start
        for line in iter(process.stdout.readline, ""):
            print(line.strip())  # Optional: print the server logs
            if "Route: /v1/embeddings, Methods: POST" in line:
                print("vLLM server is ready.")
                break
            elif "Error" in line:
                print("An error occurred during server startup.")
                process.terminate()
                raise RuntimeError("Failed to start vLLM server.")
        self.process = process

    def free_memory(self):
        """Delete the llm object and free the memory"""
        self.process.terminate()
        self.process.communicate()  # Ensures all output is flushed
        self.process.wait()  # Now safe to wait for process exit
        del self.process
        torch.cuda.empty_cache()
        gc.collect()


class EndpointLLM:
    def __init__(self):
        self.client = OpenAI(
            api_key="[EMPTY]",
            base_url="http://localhost:8001/v1",
            timeout=None,
        )

    def __call__(self, messages, *args, **kwargs):
        output = self.client.chat.completions.create(
            model="/tmp/saved_model",
            messages=messages,
            max_tokens=500,
            temperature=temperature,
        ).choices[0]
        response = output.message.content
        return messages, response


def forward_generate(prompt, context):
    """Run generations for a seed prompt and a context."""
    llm = EndpointLLM()
    messages, response = llm(
        [
            {
                "role": "user",
                "content": context + "\n\n" + prompt,
            },
        ]
    )
    return messages, response


def backward_logprobs_rewards(model, tokenizer, latents, contexts, prompts):
    """Get rewards as the log-probability of the context given the latents and prompts."""
    query_tensors = [
        torch.tensor(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": latent + "\n\n" + prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        for latent, prompt in zip(latents, prompts)
    ]
    response_tensors = [
        torch.tensor(t)
        for t in tokenizer(
            contexts,
            add_special_tokens=True,
        )["input_ids"]
    ]

    query_tensors = pad(
        query_tensors,
        tokenizer.pad_token_id,
        padding_side="left",
    ).to(device)
    response_tensors = pad(
        response_tensors,
        tokenizer.pad_token_id,
        padding_side="right",
    ).to(device)

    query_response_tensor = torch.cat((query_tensors, response_tensors), dim=1)
    log_probs = get_logprobs(
        model,
        tokenizer,
        query_tensors,
        query_response_tensor,
        response_tensors,
        batch_size=4,
        average_logprobs=True,
    )
    del query_tensors, response_tensors, query_response_tensor
    torch.cuda.empty_cache()
    return log_probs


def llm_as_a_judge_rewards(llm, latents, contexts, prompts):
    """Get rewards from the LLM as a judge."""
    pass


@torch.no_grad()
def gather_episodes(model, ref_model, tokenizer, problem_list):
    rewards = []
    logits = []
    query_tensors = []
    response_tensors = []
    prompts = []

    current_task = np.random.choice(forward_prompts, 1, replace=False)[0]

    # repeat the problems for the number of episodes
    for problem in problem_list:
        for _ in range(num_episodes_per_problem):
            prompts.append(problem)

    queries = []
    responses = []
    rewards = []

    with VLLMEndpoint(model, tokenizer) as vllm:
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(
                tqdm(
                    executor.map(partial(forward_generate, current_task), prompts),
                    total=len(prompts),
                    desc="Running episodes",
                )
            )

    # free memory of the vllm server
    torch.cuda.empty_cache()

    for p, r in results:
        queries.extend([p])
        responses.extend([r])

    # tokenization
    query_tensors = [
        torch.tensor(
            tokenizer.apply_chat_template(
                q,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        for q in queries
    ]

    response_tensors = [
        torch.tensor(t)
        for t in tokenizer(
            responses,
            add_special_tokens=True,
        )["input_ids"]
    ]

    query_tensors = pad(
        query_tensors,
        tokenizer.pad_token_id,
        padding_side="left",
    ).to(device)

    response_tensors = pad(
        response_tensors,
        tokenizer.pad_token_id,
        padding_side="right",
    ).to(device)

    query_response_tensor = torch.cat((query_tensors, response_tensors), dim=1)

    rewards = backward_logprobs_rewards(
        ref_model, tokenizer, responses, prompts, [backward_prompts[0]] * len(queries)
    )
    padding_mask = response_tensors.ne(tokenizer.pad_token_id)

    print("====================================")
    print("Episodes gathered.")
    print(f"Number of queries: {len(queries)}")
    print(f"Average reward: {rewards.mean().item()}")
    print(f"Example query: {prompts[0][:-200]}")
    print(f"Example prompt: {current_task}")
    print(f"Example response: {responses[0]}")
    print("====================================")

    return (
        rewards,
        query_tensors,
        response_tensors,
        query_response_tensor,
        padding_mask,
        None,
        queries,
        responses,
        prompts,
    )


def get_logprobs(
    model,
    tokenizer,
    query_tensors,
    query_response_tensors,
    response_tensors,
    batch_size=4,
    average_logprobs=False,
):
    logprobs = []
    context_length = query_tensors.shape[1]
    attention_mask = query_response_tensors != tokenizer.pad_token_id

    for batch in range(0, len(query_response_tensors), batch_size):
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            output = model(
                input_ids=query_response_tensors[batch : batch + batch_size],
                attention_mask=attention_mask[batch : batch + batch_size],
                return_dict=True,
            )
            logits = output.logits[:, context_length - 1 : -1]
            logits /= temperature + 1e-7
            all_logprob = torch.nn.functional.log_softmax(logits, dim=-1)
            logprob = torch.gather(
                all_logprob,
                2,
                response_tensors[batch : batch + batch_size].unsqueeze(-1),
            ).squeeze(-1)
        logprobs.append(logprob)
        del output, logits, all_logprob

    logprobs = torch.cat(logprobs, 0)
    if average_logprobs:
        response_mask = response_tensors.ne(tokenizer.pad_token_id)
        logprobs = (logprobs * response_mask).sum(dim=1) / response_mask.sum(dim=1)
        del response_mask

    torch.cuda.empty_cache()
    return logprobs


class AccumulatorDict:
    """Accumulate values in a dictionary."""

    def __init__(self):
        self.data = {}

    def accumulate(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def mean(self, key):
        return sum(self.data[key]) / len(self.data[key])

    def get(self):
        # return mean values and clear the stats
        data = {k: sum(v) / len(v) for k, v in self.data.items()}
        self.data = {}
        return data

    def clear(self):
        self.data = {}


class PPONoValueTrainer:
    """PPO without a value function."""

    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stats = AccumulatorDict()

    @torch.no_grad()
    def generate_episodes(self, problem_list):
        import numpy as np

        (
            rewards,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
            logprob,
            queries,
            responses,
            prompts,
        ) = gather_episodes(self.model, self.ref_model, self.tokenizer, problem_list)
        torch.cuda.empty_cache()

        if logprob is None:
            print("Computing logprobs under current policy.")
            logprob = get_logprobs(
                self.model,
                self.tokenizer,
                query_tensors,
                query_response_tensors,
                response_tensors,
            )

        # move ref model to device to avoid OOM
        self.ref_model.to("cuda")

        context_length = query_tensors.shape[1]
        ref_logprob = get_logprobs(
            self.ref_model,
            self.tokenizer,
            query_tensors,
            query_response_tensors,
            response_tensors,
        )

        # move ref model to cpu to avoid OOM
        self.ref_model.to("cpu")

        torch.cuda.empty_cache()
        advantages = masked_whiten(
            rewards.unsqueeze(1), padding_mask, unbiased_variance=False
        )
        torch.cuda.empty_cache()

        return (
            rewards,
            advantages,
            ref_logprob,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
        ), (queries, responses, prompts)

    def compute_loss(self, episode_returns) -> float:
        (
            mb_rewards,
            mb_advantage,
            mb_logprobs,  # values of the previous policy to compute the importance ratio
            mb_ref_logprobs,  # values of the reference policy
            mb_queries,
            mb_responses,
            mb_query_responses,
            mb_padding_mask,
        ) = episode_returns
        context_length = mb_queries.shape[1]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            output = self.model(
                input_ids=mb_query_responses,
                attention_mask=mb_query_responses.ne(self.tokenizer.pad_token_id),
                return_dict=True,
            )
            logits = output.logits[:, context_length - 1 : -1]
            logits /= temperature + 1e-7
            new_all_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(
                new_all_logprobs, 2, mb_responses.unsqueeze(-1)
            ).squeeze(-1)

            log_ratio = (new_logprobs - mb_logprobs) * mb_padding_mask
            ratio = torch.exp(log_ratio)
            pg_losses1 = -mb_advantage * ratio
            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.2, 1.2)
            pg_losses = torch.max(pg_losses1, pg_losses2)
            pg_loss = (pg_losses * mb_padding_mask).sum() / mb_padding_mask.sum()

            log_ratio = mb_ref_logprobs - new_logprobs
            kl = torch.exp(log_ratio) - log_ratio - 1
            ref_kl = torch.clamp(
                kl * mb_padding_mask,
                min=0,
                max=10,
            )
            ref_kl_loss = kl_coeff * ref_kl.sum(dim=1).mean()

            self.stats.accumulate("ref_kl_loss", ref_kl.sum(1).mean().item())
            self.stats.accumulate("pg_loss", pg_loss.item())
            self.stats.accumulate("mb_rewards", mb_rewards.mean().item())

            return pg_loss + ref_kl_loss
            del (
                logits,
                mb_rewards,
                mb_advantage,
                mb_query_responses,
                mb_logprobs,
                mb_responses,
            )


class RLOOTrainer(PPONoValueTrainer):
    @torch.no_grad()
    def generate_episodes(self, problem_list):
        import numpy as np

        (
            rewards,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
            logprob,
            queries,
            responses,
            prompts,
        ) = gather_episodes(self.model, self.ref_model, self.tokenizer, problem_list)

        torch.cuda.empty_cache()

        if logprob is None:
            logprob = get_logprobs(
                self.model,
                self.tokenizer,
                query_tensors,
                query_response_tensors,
                response_tensors,
            )

        ref_logprob = get_logprobs(
            self.ref_model,
            self.tokenizer,
            query_tensors,
            query_response_tensors,
            response_tensors,
        )

        torch.cuda.empty_cache()

        # rloo rewards
        rewards = rewards.reshape(-1, num_episodes_per_problem)
        baseline = (rewards.sum(1).unsqueeze(1) - rewards) / (
            num_episodes_per_problem - 1
        )
        advantages = (rewards - baseline).view(-1, 1)

        torch.cuda.empty_cache()

        return (
            rewards,
            advantages,
            logprob,
            ref_logprob,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
        ), (queries, responses, prompts)


class PEFTRLOOTrainer(RLOOTrainer):
    def __init__(self, model_name):
        self.model = get_peft_model(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stats = AccumulatorDict()

    @torch.no_grad()
    def generate_episodes(self, problem_list):
        import numpy as np

        (
            rewards,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
            logprob,
            queries,
            responses,
            prompts,
        ) = gather_episodes(self.model, self.ref_model, self.tokenizer, problem_list)

        torch.cuda.empty_cache()

        if logprob is None:
            logprob = get_logprobs(
                self.model,
                self.tokenizer,
                query_tensors,
                query_response_tensors,
                response_tensors,
            )

        ref_logprob = get_logprobs(
            self.ref_model,
            self.tokenizer,
            query_tensors,
            query_response_tensors,
            response_tensors,
        )

        torch.cuda.empty_cache()

        # rloo rewards
        rewards = rewards.reshape(-1, num_episodes_per_problem)
        baseline = (rewards.sum(1).unsqueeze(1) - rewards) / (
            num_episodes_per_problem - 1
        )
        advantages = (rewards - baseline).view(-1, 1)

        torch.cuda.empty_cache()

        # create no_context tensors
        prompts = [
            torch.tensor(
                self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": forward_prompts[0],
                        },
                    ],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            )
            for _ in range(len(responses))
        ]
        nc_query_tensors = pad(
            prompts,
            self.tokenizer.pad_token_id,
            padding_side="left",
        ).to(device)
        nc_query_response_tensors = torch.cat(
            (nc_query_tensors, response_tensors), dim=1
        )

        return (
            rewards,
            advantages,
            logprob,
            ref_logprob,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
            nc_query_tensors,
            nc_query_response_tensors,
        ), (queries, responses, prompts)

    def compute_loss(self, episode_returns) -> float:
        (
            mb_rewards,
            mb_advantage,
            mb_logprobs,  # values of the previous policy to compute the importance ratio
            mb_ref_logprobs,  # values of the reference policy
            mb_queries,
            mb_responses,
            mb_query_responses,
            mb_padding_mask,
            mb_nc_query_tensors,
            mb_nc_query_response_tensors,
        ) = episode_returns
        context_length = mb_queries.shape[1]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            output = self.model(
                input_ids=mb_query_responses,
                attention_mask=mb_query_responses.ne(self.tokenizer.pad_token_id),
                return_dict=True,
            )
            logits = output.logits[:, context_length - 1 : -1]
            logits /= temperature + 1e-7
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs = torch.gather(logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)

            log_ratio = (logprobs - mb_logprobs) * mb_padding_mask
            ratio = torch.exp(log_ratio)
            pg_losses1 = -mb_advantage * ratio
            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.2, 1.2)
            pg_losses = torch.max(pg_losses1, pg_losses2)
            pg_loss = (pg_losses * mb_padding_mask).sum() / mb_padding_mask.sum()

            log_ratio = mb_ref_logprobs - logprobs
            kl = torch.exp(log_ratio) - log_ratio - 1
            ref_kl = torch.clamp(
                kl * mb_padding_mask,
                min=0,
                max=10,
            )
            ref_kl_loss = kl_coeff * ref_kl.sum(dim=1).mean()
            del (
                logits,
                mb_advantage,
                mb_query_responses,
                logprobs,
                mb_logprobs,
                mb_ref_logprobs,
            )

            # data distillation
            context_length = mb_nc_query_tensors.shape[1]
            output = self.model(
                input_ids=mb_nc_query_response_tensors,
                attention_mask=mb_nc_query_response_tensors.ne(
                    self.tokenizer.pad_token_id
                ),
                return_dict=True,
            )
            logprobs = torch.nn.functional.log_softmax(
                output.logits[:, context_length - 1 : -1], dim=-1
            )
            logprobs = torch.gather(logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
            km_loss = (-logprobs * mb_padding_mask).sum() / mb_padding_mask.sum()

            del (
                mb_responses,
                mb_nc_query_response_tensors,
                mb_nc_query_tensors,
                logprobs,
            )
            torch.cuda.empty_cache()

            self.stats.accumulate("km_loss", km_loss.item())
            self.stats.accumulate("ref_kl_loss", ref_kl.sum(1).mean().item())
            self.stats.accumulate("pg_loss", pg_loss.item())
            self.stats.accumulate("mb_rewards", mb_rewards.mean().item())
            return pg_loss + ref_kl_loss + km_loss

    @property
    def ref_model(self):
        class RefWrapper:
            def __init__(self, model):
                self.model = model

            def __call__(self, *args, **kwargs):
                with disable_modifiers(self.model):
                    return self.model.forward(*args, **kwargs)

        return RefWrapper(self.model)


class RFTTrainer:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate_episodes(self, problem_list):
        import numpy as np

        (
            rewards,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
            logprob,
            queries,
            responses,
            prompts,
        ) = gather_episodes(self.model, self.model, self.tokenizer, problem_list)

        torch.cuda.empty_cache()

        # rloo rewards
        rewards = rewards.reshape(-1, num_episodes_per_problem)
        N, M = rewards.shape

        # Sort to get descending order indices
        _, sorted_idx = torch.sort(rewards, dim=1, descending=True)

        # Create the rank values (1.0, (M-1)/M, ..., 1/M)
        rank_values = (
            torch.arange(M, 0, -1, dtype=rewards.dtype, device=rewards.device) / M
        )

        # Scatter rank values back to original order
        advantages = torch.zeros_like(rewards)
        advantages.scatter_(1, sorted_idx, rank_values.unsqueeze(0).expand(N, M))
        advantages = advantages.reshape(-1)
        torch.cuda.empty_cache()

        return (
            rewards,
            advantages,
            query_tensors,
            response_tensors,
            query_response_tensors,
            padding_mask,
        ), (queries, responses, prompts)

    def compute_loss(self, episode_returns) -> float:
        (
            mb_rewards,
            mb_advantage,
            mb_queries,
            mb_responses,
            mb_query_responses,
            mb_padding_mask,
        ) = episode_returns
        context_length = mb_queries.shape[1]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            output = self.model(
                mb_query_responses,
                attention_mask=mb_query_responses.ne(self.tokenizer.pad_token_id),
                return_dict=True,
            )
            logits = output.logits[:, context_length - 1 : -1]
            logits /= temperature + 1e-7
            new_all_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(
                new_all_logprobs, 2, mb_responses.unsqueeze(-1)
            ).squeeze(-1)

            loss = (-new_logprobs * mb_padding_mask) * mb_advantage.unsqueeze(1)
            loss = loss.sum() / mb_padding_mask.sum()
            return loss


def eval_generative_qa(model, tokenizer, dataset):
    from mttl.dataloader.ni_metrics import compute_metrics

    # create array of all questions in dataset
    questions = []
    for i in range(len(dataset)):
        questions.extend(dataset[i]["questions"])
    # same for answers
    answers = []
    for i in range(len(dataset)):
        answers.extend(dataset[i]["answers"])
    # now tokenize and pad
    question_tensors = [
        torch.tensor(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Answer the following question briefly and to the point: "
                        + question,
                    }
                ],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        for question in questions
    ]
    # don't do it for answers, we don't need them
    question_tensors = pad(
        question_tensors,
        tokenizer.pad_token_id,
        padding_side="left",
    ).to(device)
    questions_mask = question_tensors.ne(tokenizer.pad_token_id)
    # now generate answers with a given batch size
    with torch.no_grad():
        generations = []
        # use tqdm instead of range
        for i in tqdm(range(0, len(question_tensors), batch_size)):
            output = model.generate(
                input_ids=question_tensors[i : i + batch_size],
                attention_mask=questions_mask[i : i + batch_size],
                do_sample=True,
                max_length=500,
                temperature=temperature,
                num_return_sequences=1,
            )
            generations.extend(output[:, question_tensors.shape[1] :].cpu().tolist())
    # now we have the generations, let's detokenize them
    generations = [tokenizer.decode(g, skip_special_tokens=True) for g in generations]
    # now compute multi-reference rouge
    return compute_metrics(generations, answers)["rougeL"]


def get_wsd_scheduler(
    optimizer,
    total_steps,
    min_lr_ratio=0.0,
    last_epoch=-1,
):
    from torch.optim.lr_scheduler import LambdaLR

    # Calculate step counts from proportions
    num_warmup_steps = int(total_steps * 0.01)
    num_stable_steps = int(total_steps * 0.89)
    num_decay_steps = total_steps - num_warmup_steps - num_stable_steps

    def get_ratio(
        current_step, num_warmup_steps, num_stable_steps, num_decay_steps, min_lr_ratio
    ):
        if current_step < num_warmup_steps:
            return (
                current_step / float(num_warmup_steps) if num_warmup_steps > 0 else 1.0
            )
        elif current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        elif current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
            steps_in_decay = num_warmup_steps + num_stable_steps
            progress = (
                (current_step - steps_in_decay) / float(num_decay_steps)
                if num_decay_steps > 0
                else 1.0
            )
            return 1.0 - (1.0 - min_lr_ratio) * progress
        return min_lr_ratio

    def lr_lambda(step):
        return get_ratio(
            step,
            num_warmup_steps,
            num_stable_steps,
            num_decay_steps,
            min_lr_ratio,
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "q1.5": "Qwen/Qwen2.5-1.5B-Instruct",
}

num_problems_per_step = 5  # number of problems per step
num_episodes_per_problem = 5  # number of episodes per problem
batch_size = 4  # batch size
num_ppo_epochs = 4  # number of off-policy epochs
acc_batch_size = 1  # accumulation batch size
num_steps = 100  # num total steps (updates)
temperature = 0.5  # temperature for the LLM
kl_coeff = 1e-4  # kl coefficient

# Get the model name
for mn in models.keys():
    if f"-{mn}" in sys.argv:
        model_name = models[mn]
        break

# Set number of inner ppo off policy epochs
if "-ppo_epochs" in sys.argv:
    num_ppo_epochs = int(sys.argv[sys.argv.index("-ppo_epochs") + 1])

# Trainer
if "-rloo" in sys.argv:
    algo = RLOOTrainer(model_name)
elif "-rft" in sys.argv:
    algo = RFTTrainer(model_name)
elif "-peft" in sys.argv:
    algo = PEFTRLOOTrainer(model_name)
else:
    raise ValueError("Invalid algorithm type.")

# lr
if "-lr" in sys.argv:
    lr = float(sys.argv[sys.argv.index("-lr") + 1])
else:
    lr = 1e-5

# document id
if "-task" in sys.argv:
    task = sys.argv[sys.argv.index("-task") + 1]
else:
    raise ValueError("Task not specified.")

# output directory
if "-o" in sys.argv:
    import shutil

    output_dir = sys.argv[sys.argv.index("-o") + 1]
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = pjoin(output_dir, task)
    os.makedirs(output_dir, exist_ok=True)
else:
    raise ValueError("Output directory not specified.")

dataset = load_dataset("sordonia/narrativeqa_sanitized", split="train").filter(
    lambda x: x["document_id"] == task
)
chunks = list(chunk_text(dataset["text"][0], algo.tokenizer))
if num_problems_per_step == -1:
    num_problems_per_step = len(chunks)

# set params in optimized group as trainable and others as not
optimized_group = []
for n, p in algo.model.named_parameters():
    p.requires_grad = "lora" in n
    if p.requires_grad:
        optimized_group.append(p)
# print number of trainable parameters
print(
    f"Number of trainable parameters: {sum(p.numel() for p in optimized_group)}",
)
# optimizer
optimizer = torch.optim.Adam(optimized_group, lr=lr)
# get scheduler now
if "-wsd" in sys.argv:
    scheduler = get_wsd_scheduler(
        optimizer,
        total_steps=num_steps
        * num_ppo_epochs
        * (num_problems_per_step * num_episodes_per_problem)
        // batch_size,
    )
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

training_stats = []
for epoch in range(num_steps):
    rouge_l = eval_generative_qa(algo.model, algo.tokenizer, dataset)
    episode_data, generations = algo.generate_episodes(
        list(np.random.choice(chunks, num_problems_per_step, replace=False))
    )

    torch.save(episode_data, f"{output_dir}/episode_data_{epoch}.pt")
    torch.save(generations, f"{output_dir}/generations_{epoch}.pt")

    avg_reward = episode_data[0].mean().item()
    print(f"Epoch {epoch}, avg reward: {avg_reward}")

    for ppo_epoch_idx in range(num_ppo_epochs):
        dataset_size = len(episode_data[0])
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        epoch_stats = AccumulatorDict()

        # Iterate over the dataset in batches
        for start_idx in range(0, dataset_size, batch_size):
            b_inds = indices[start_idx : start_idx + batch_size]

            loss_batch = 0
            for step in range(0, len(b_inds), acc_batch_size):
                mini_batch_inds = b_inds[step : step + acc_batch_size]
                loss = algo.compute_loss([x[mini_batch_inds] for x in episode_data])
                loss = loss / batch_size
                loss.backward()
                loss_batch += loss.item()
                del loss

            epoch_stats.accumulate("loss", loss_batch)
            torch.nn.utils.clip_grad_norm_(algo.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        print(
            f"Epoch {epoch}, {algo.__class__.__name__}, off-epoch: {ppo_epoch_idx}, loss: {epoch_stats.mean('loss')}, lr: {scheduler.get_last_lr()[0]}"
        )

    # append a bunch of training stats
    training_stats.append(
        {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "avg_reward": avg_reward,
            "rouge_l": rouge_l,
            **epoch_stats.get(),
            **algo.stats.get(),
        }
    )
    # save the training stats
    torch.save(training_stats, f"{output_dir}/training_stats.pt")
