from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.kms.deductron.algos.algo import Algo, Request, RequestUtils
from projects.kms.deductron.data_utils import create_joint_tensors
from projects.kms.deductron.ddp_utils import rank_zero_only, ddp_state
from projects.kms.deductron.gen_utils import GenerationBackend
from projects.kms.deductron.sgl_utils import SGLGenerator
from projects.kms.deductron.utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    compute_kl_divergence,
    flatten,
    get_logprobs,
    print_metrics,
    repeat,
    get_shifted_logprobs,
    masked_mean,
    masked_whiten,
)
from projects.kms.deductron.ddp_utils import ddp_state


class RLOO(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl",
            type=float,
            default=0.0001,
            help="Target KL divergence between policy and reference policy.",
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
        self.k = k
        self.task = task
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.kl_ctl = algo_kwargs.get("kl_ctl", 0.0001)
        self.stats = AccumulatorDict()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        self.ref_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        self.ref_model.to(ddp_state.device)

        rewards = task.get_rewards(
            model=self.ref_model,
            tokenizer=self.tokenizer,
            requests=evaluation_requests,
            temperature=self.temperature,
        )

        RequestUtils.populate(evaluation_requests, rewards, "reward")
        return evaluation_requests

    @torch.no_grad()
    def gather_episodes(
        self,
        prompts: List[str],
        labels: List[str],
    ):
        self.ref_model.to(ddp_state.device)
        evaluation_requests = self.compute_rewards(prompts, labels, k=self.k)
        rewards = [r.reward for r in evaluation_requests]
        finished = [r.finished for r in evaluation_requests]
        messages = [r.messages for r in evaluation_requests]
        responses = [r.response for r in evaluation_requests]

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)

        self.stats.accumulate("avg_reward", avg_reward)
        self.stats.accumulate("max_reward", max_reward)
        self.stats.accumulate("finished", (100.0 * np.sum(finished) / len(finished)))

        query_response_tensors, query_response_mask, response_mask = (
            create_joint_tensors(
                self.tokenizer,
                messages,
                responses,
                finished,
                max_length=4096,
                pad_to_length=4096,
            )
        )

        ref_logprobs = get_logprobs(
            self.ref_model,
            query_response_tensors,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )
        self.ref_model.to("cpu")

        self.model.eval()
        old_logprobs = get_logprobs(
            self.model,
            query_response_tensors,
            query_response_mask,
            response_mask,
            temperature=self.temperature,
            reduction="none",
        )
        self.model.train()

        # trl adds this to the rewards
        action_mask = response_mask[:, 1:]
        kl = old_logprobs - ref_logprobs
        kl_loss = (kl * action_mask).sum(1)
        kl_rewards = (-kl_loss).tolist()

        self.stats.accumulate("kl_loss", kl_loss.mean().item())
        self.stats.accumulate("kl_reward", np.mean(kl_rewards))

        rewards = [r + self.kl_ctl * nr for r, nr in zip(rewards, kl_rewards)]
        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, self.k)
        baseline = (rewards.sum(1).unsqueeze(-1) - rewards) / (self.k - 1)
        advantages = (rewards - baseline).view(-1, 1)

        if ddp_state.is_main_process:
            print("====================================")
            print("Optimistic reward:", max_reward)
            print("Average reward:", avg_reward)
            print(
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
            print("Finished: ", (100.0 * np.sum(finished)) / len(finished), "%")
            print("====================================")

            print("====================================")
            print("Response 0 > Response -1:")
            sorted_idx = torch.argsort(rewards[:5].flatten(), descending=True)
            best_idx = sorted_idx[0].item()
            last_idx = sorted_idx[-1].item()
            print(
                f"Response 0, Reward {rewards[:5].flatten()[best_idx].item():.4f}:\n{responses[best_idx]}"
            )
            print("------------------------------------")
            print(
                f"Response -1, Reward {rewards[:5].flatten()[last_idx].item():.4f}:\n{responses[last_idx]}"
            )

        outputs = (
            query_response_tensors,
            query_response_mask,
            response_mask,
            advantages,
            old_logprobs,
        )
        return outputs

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
            mb_advantage,
            mb_old_logprobs,
        ) = episode_returns

        # trim from mb_query_response everything that is after the last 1 token in mb_query_response_mask
        max_tokens = mb_query_response_mask.sum(dim=1).max()
        mb_query_response = mb_query_response[:, :max_tokens]
        mb_query_response_mask = mb_query_response_mask[:, :max_tokens]
        mb_response_mask = mb_response_mask[:, :max_tokens]
        mb_old_logprobs = mb_old_logprobs[:, : max_tokens - 1]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
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

        pg_losses1 = -mb_advantage * ratio
        pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.2, 1.2)
        pg_losses = torch.max(pg_losses1, pg_losses2)
        pg_loss = masked_mean(pg_losses, action_mask)

        del (
            output,
            mb_logprobs,
            mb_query_response,
            mb_old_logprobs,
            mb_query_response_mask,
            mb_response_mask,
            action_mask,
        )
        return pg_loss
