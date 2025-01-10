from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from projects.kms.deductron.algos.algo import Algo, Request, RequestUtils
from projects.kms.deductron.data_utils import create_joint_tensors
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
from accelerate.state import AcceleratorState
from dataclasses import dataclass


@dataclass
class PriRequest(Request):
    prior_reward: float = None


class RFT(Algo):
    @classmethod
    def add_parser_args(self, parser):
        parser.add_argument(
            "--kl_ctl",
            type=float,
            default=0.001,
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
        self.acc_state = AcceleratorState()
        self.stats = AccumulatorDict()

        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        self.ref_model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        self.use_prior = True
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
                PriRequest(
                    query_id=query_id,
                    messages=query_messages,
                    response=query_response,
                    label=label,
                    finished=finish,
                )
            )

        self.ref_model.to(self.acc_state.device)
        rewards = task.get_rewards(
            model=self.ref_model,
            tokenizer=self.tokenizer,
            requests=evaluation_requests,
            temperature=self.temperature,
        )

        qr, qrm, rm = create_joint_tensors(
            self.tokenizer,
            [r.messages for r in evaluation_requests],
            [r.response for r in evaluation_requests],
            [r.finished for r in evaluation_requests],
            max_length=4096,
            pad_to_length=4096,
        )
        log_probs = (
            get_logprobs(
                self.ref_model,
                qr,
                qrm,
                rm,
                batch_size=2,
                temperature=self.temperature,
            )
            .cpu()
            .tolist()
        )
        del qr, qrm, rm
        torch.cuda.empty_cache()
        self.ref_model.to("cpu")

        RequestUtils.populate(evaluation_requests, rewards, "reward")
        RequestUtils.populate(evaluation_requests, log_probs, "prior_reward")
        return evaluation_requests

    @torch.no_grad()
    def gather_episodes(
        self,
        prompts: List[str],
        labels: List[str],
    ):
        evaluation_requests = self.compute_rewards(prompts, labels, k=self.k)

        rewards = [r.reward for r in evaluation_requests]
        finished = [r.finished for r in evaluation_requests]
        pri_rewards = [r.prior_reward for r in evaluation_requests]

        max_reward, avg_reward = RequestUtils.gather_max_avg_reward(evaluation_requests)
        pri_max_reward, pri_avg_reward = RequestUtils.gather_max_avg_reward(
            evaluation_requests, "prior_reward"
        )

        self.stats.accumulate("avg_reward", avg_reward)
        self.stats.accumulate("max_reward", max_reward)
        self.stats.accumulate("pri_max_reward", pri_max_reward)
        self.stats.accumulate("pri_avg_reward", pri_avg_reward)
        self.stats.accumulate("finished", (100.0 * np.sum(finished) / len(finished)))

        if self.acc_state.is_main_process:
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
            sorted_idx = np.argsort(rewards[:5])[::-1]
            best_idx = sorted_idx[0]
            last_idx = sorted_idx[-1]
            print(
                f"Response 0, Reward {rewards[best_idx]:.4f}:\n{evaluation_requests[best_idx].response}"
            )
            print("------------------------------------")
            print(
                f"Response -1, Reward {rewards[last_idx]:.4f}:\n{evaluation_requests[last_idx].response}"
            )

        # pick the best response for each query
        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(-1, self.k)
        pri_rewards = torch.tensor(pri_rewards, dtype=torch.float32).reshape(-1, self.k)

        messages = []
        responses = []
        finished = []
        for i, rew in enumerate(rewards):
            max_reward_idx = torch.argmax(rew + pri_rewards[i]).item()
            messages.append(evaluation_requests[i * self.k + max_reward_idx].messages)
            responses.append(evaluation_requests[i * self.k + max_reward_idx].response)
            finished.append(evaluation_requests[i * self.k + max_reward_idx].finished)

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

        outputs = (
            query_response_tensors,
            query_response_mask,
            response_mask,
        )
        return outputs

    def compute_loss(self, episode_returns) -> float:
        (
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
        ) = episode_returns

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
        mb_logprobs = torch.gather(shift_logits, 2, shift_labels.unsqueeze(-1)).squeeze(
            -1
        )

        loss = -mb_logprobs
        loss = (loss * shift_labels_mask).sum() / shift_labels_mask.sum()

        del (
            shift_logits,
            shift_labels,
            shift_labels_mask,
            mb_query_response,
            mb_query_response_mask,
            mb_response_mask,
            logits,
            output,
        )
        torch.cuda.empty_cache()
        return loss
