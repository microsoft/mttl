from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.kms.deductron.algos.algo import Algo, Request, RequestUtils
from projects.kms.deductron.data_utils import create_joint_tensors
from projects.kms.deductron.ddp_utils import rank_zero_only
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
)
from projects.kms.deductron.vllm_utils import VLLMGenerator


class RLOO(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl",
            type=float,
            default=0.0,
            help="Target KL divergence between policy and reference policy.",
        )
        return parser

    def __init__(
        self,
        model_name,
        k=5,
        temperature=DEFAULT_TEMP,
        max_tokens=DEFAULT_MAX_TOKENS,
        task_generator="summary",
        reward_func="infogain",
        device="cuda",
        **algo_kwargs,
    ):
        self.k = k
        self.reward_func = reward_func
        self.task_generator = task_generator
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.kl_ctl = algo_kwargs.get("kl_ctl", 0.0)
        self.stats = AccumulatorDict()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

    @rank_zero_only
    @torch.no_grad()
    def gather_episodes(
        self,
        prompts: List[str],
        labels: List[str],
    ):
        vllm = GenerationBackend.get()

        if self.task_generator == "summary":
            from projects.kms.deductron.algos.utils import summary_task_generator

            messages = summary_task_generator(prompts)
        else:
            raise ValueError(f"Unknown task generator: {self.task_generator}")

        # Gather first set of completions
        responses, finished = vllm.chat(
            messages,
            temperature=self.temperature,
            n=self.k,
            max_tokens=self.max_tokens,
            return_finished=True,
        )

        responses = flatten(responses)
        finished = flatten(finished)
        query_ids = repeat(range(len(messages)), self.k)
        labels = repeat(labels, self.k)
        messages = repeat(messages, self.k)

        print("====================================")
        print("Example response:")
        print(messages[0])
        print(responses[0])

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

        if self.reward_func == "infogain":
            from .utils import infogain_reward

            rewards = infogain_reward(
                model=self.ref_model,
                tokenizer=self.tokenizer,
                messages=[r.messages for r in evaluation_requests],
                responses=[r.response for r in evaluation_requests],
                labels=[r.label for r in evaluation_requests],
            )
        else:
            raise ValueError(f"Unknown reward function: {self.reward_func}")

        RequestUtils.populate(evaluation_requests, rewards, "reward")

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)

        self.stats.accumulate("avg_reward", avg_reward)
        self.stats.accumulate("max_reward", max_reward)
        self.stats.accumulate("finished", (100.0 * np.sum(finished) / len(finished)))

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

        # rloo advantages
        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, self.k)
        max_rewards = torch.max(rewards, dim=1).values
        baseline = (rewards.sum(1).unsqueeze(1) - rewards) / (self.k - 1)
        advantages = (rewards - baseline).view(-1, 1)

        queries = []
        responses = []
        for request in evaluation_requests:
            queries.append(request.messages)
            responses.append(request.response)

        query_response_tensors, query_response_mask, response_mask = (
            create_joint_tensors(
                self.tokenizer,
                queries,
                responses,
            )
        )

        outputs = (
            query_response_tensors,
            query_response_mask,
            response_mask,
            advantages,
        )

        if self.kl_ctl:
            ref_logprobs = get_logprobs(
                self.ref_model,
                query_response_tensors,
                query_response_mask,
                response_mask,
                temperature=self.temperature,
                reduction="none",
            )
            outputs += (ref_logprobs,)

        return outputs

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
            mb_advantage,
        ) = episode_returns[:4]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            # trim from mb_query_response everything that is after the last 1 token in mb_query_response_mask
            max_tokens = mb_query_response_mask.sum(dim=1).max()

            mb_query_response = mb_query_response[:, :max_tokens]
            mb_query_response_mask = mb_query_response_mask[:, :max_tokens]
            mb_response_mask = mb_response_mask[:, :max_tokens]

            output = self.model(
                input_ids=mb_query_response,
                attention_mask=mb_query_response_mask,
                return_dict=True,
            )
            logits = output.logits / (self.temperature + 1e-7)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

            shift_logits = logits[:, :-1]
            shift_labels = mb_query_response[:, 1:]
            shift_labels_mask = mb_response_mask[:, 1:]
            mb_logprobs = torch.gather(
                shift_logits, 2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            loss = -mb_logprobs * (mb_advantage * (mb_advantage > 0)).float()
            loss = (loss * shift_labels_mask).sum() / shift_labels_mask.sum()

            if self.kl_ctl:
                mb_ref_logprobs = episode_returns[-1]
                mb_ref_logprobs = mb_ref_logprobs[:, : max_tokens - 1]
                ref_kl = compute_kl_divergence(
                    mb_ref_logprobs, mb_logprobs, mask=shift_labels_mask
                )
                loss += self.kl_ctl * ref_kl
                self.stats.accumulate("kl_loss", ref_kl.item())
                del mb_ref_logprobs

            del (
                mb_logprobs,
                shift_logits,
                shift_labels,
                shift_labels_mask,
                logits,
                output,
            )
            torch.cuda.empty_cache()
            return loss
