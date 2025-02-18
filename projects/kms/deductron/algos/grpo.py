from collections import defaultdict
from typing import Dict, List, Union

import torch
import copy
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from ddp_utils import rank_zero_only, ddp_state
from utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    get_logprobs,
    get_shifted_logprobs,
    masked_mean,
    masked_whiten,
    print_metrics,
    repeat,
    flatten,
    compute_kl_divergence,
)

from projects.kms.deductron.data_utils import create_joint_tensors, get_ending_tokens, pad_query_and_response
from projects.kms.deductron.gen_utils import GenerationBackend
from projects.kms.deductron.algos.algo import Algo, Request, RequestUtils
from projects.kms.deductron.algos.rft import RFT


class GRPO(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl", type=float, default=0.0001, help="KL term control"
        )
        parser.add_argument(
            "--kl_max", type=int, default=10, help="Clip KL divergence to this value"
        )
        return parser

    def __init__(
        self,
        model_name,
        k=5,
        temperature=DEFAULT_TEMP,
        max_tokens=DEFAULT_MAX_TOKENS,
        task="summary_autoencoder",
        device="cuda",
        **algo_kwargs,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.task = task
        self.k = k
        self.kl_ctl = algo_kwargs["kl_ctl"]
        self.kl_max = algo_kwargs["kl_max"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stats = AccumulatorDict()

    def compute_rewards(
        self,
        prompts: List[str],
        labels: List[str],
        k=1,
    ) -> List[Request]:
        from .task import get_task

        vllm = GenerationBackend.get()

        task = get_task(self.task)
        messages = task.encode_template(prompts)

        # Gather first set of completions
        responses, finished = vllm.chat(
            messages,
            temperature=self.temperature,
            n=k,
            max_tokens=self.max_tokens,
            return_finished=True,
        )

        responses = flatten(responses)
        finished = flatten(finished)
        query_ids = repeat(range(len(messages)), k)
        labels = repeat(labels, k)
        messages = repeat(messages, k)

        evaluation_requests = []
        for query_id, query_messages, query_response, label, finish in zip(
            query_ids, messages, responses, labels, finished
        ):
            evaluation_requests.append(
                Request(
                    query_id=query_id,
                    messages=query_messages,
                    response=query_response,
                    label=label,
                    finished=finish,
                )
            )

        rewards = task.get_rewards(
            model=self.ref_model,
            tokenizer=self.tokenizer,
            requests=evaluation_requests,
            temperature=self.temperature,
        )

        RequestUtils.populate(evaluation_requests, rewards, "reward")
        return evaluation_requests

    @torch.no_grad
    def gather_episodes(self, messages, labels):
        evaluation_requests = self.compute_rewards(messages, labels, k=self.k)
        messages = [r.messages for r in evaluation_requests]
        finished = [r.finished for r in evaluation_requests]
        responses = [r.response for r in evaluation_requests]
        rewards = [r.reward for r in evaluation_requests]

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)
        self.stats.accumulate("avg_reward", avg_reward)

        ddp_state.print("====================================")
        ddp_state.print("Problem: ", evaluation_requests[0].messages[-1]["content"])
        ddp_state.print("Answer: ", evaluation_requests[0].response)
        ddp_state.print("Reward: ", evaluation_requests[0].reward)

        ddp_state.print("====================================")
        ddp_state.print("Average reward:", avg_reward)
        ddp_state.print(
            "Reward distribution:",
            print_metrics(
                [
                    np.mean(rewards)
                    for rewards in RequestUtils.group_by_query_id(
                        evaluation_requests, "reward"
                    ).values()
                ]
            ),
        )
        ddp_state.print("Finished: ", (100.0 * np.sum(finished)) / len(finished), "%")
        ddp_state.print("====================================")

        rewards = np.asarray(rewards).reshape(-1, self.k)
        advantages = (rewards - rewards.mean(axis=1, keepdims=True)) / (
            rewards.std(axis=1, keepdims=True) + 1e-8
        )
        advantages = advantages.reshape(-1).tolist()

        query_response, query_response_mask, response_mask = create_joint_tensors(
            self.tokenizer,
            messages,
            responses,
            is_final=finished,
        )
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # probabilities under the reference policy
        ref_logprobs = get_logprobs(
            self.ref_model,
            query_response,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )

        # probabilities under the sampling policy
        old_logprobs = get_logprobs(
            self.model,
            query_response,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )

        return (
            query_response,
            query_response_mask,
            response_mask,
            advantages,
            ref_logprobs,
            old_logprobs,
        )

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
            mb_advantage,
            mb_ref_logprobs,
            mb_old_logprobs,
        ) = episode_returns

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            # trim from mb_query_response everything that is after the last 1 token in mb_query_response_mask
            max_tokens = mb_query_response_mask.sum(dim=1).max()
            mb_query_response = mb_query_response[:, :max_tokens]
            mb_query_response_mask = mb_query_response_mask[:, :max_tokens]
            mb_response_mask = mb_response_mask[:, :max_tokens]
            # already shifted
            mb_ref_logprobs = mb_ref_logprobs[:, : max_tokens - 1]
            mb_old_logprobs = mb_old_logprobs[:, : max_tokens - 1]

            output = self.model(
                input_ids=mb_query_response,
                attention_mask=mb_query_response_mask,
                return_dict=True,
            )

            mb_logprobs = get_shifted_logprobs(
                output.logits,
                mb_query_response,
                mb_response_mask,
                temperature=self.temperature,
            )

            # Compute the PPO-clip loss
            action_mask = mb_response_mask[:, 1:]
            log_ratio = (mb_logprobs - mb_old_logprobs) * action_mask
            ratio = torch.exp(log_ratio)

            pg_losses1 = -mb_advantage.unsqueeze(1) * ratio
            pg_losses2 = -mb_advantage.unsqueeze(1) * torch.clamp(ratio, 0.8, 1.2)
            pg_losses = torch.max(pg_losses1, pg_losses2)

            per_token_kl = (
                torch.exp(mb_ref_logprobs - mb_logprobs)
                - (mb_ref_logprobs - mb_logprobs)
                - 1
            )
            pg_losses = pg_losses + self.kl_ctl * per_token_kl
            pg_loss = masked_mean(pg_losses, action_mask)

            self.stats.accumulate(
                "kl_loss", masked_mean(per_token_kl, action_mask).item()
            )
            return pg_loss
