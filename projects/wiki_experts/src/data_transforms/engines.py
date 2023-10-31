from typing import List
import numpy as np
import tqdm
import time
from dataclasses import dataclass, field

from vllm import SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from mttl.models.openai import GPT
from src.data_transforms.utils import INVALID_RESPONSE
from mttl.vllm_engines.engines import LLMEngine


@dataclass
class Response:
    contexts: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    cumulative_logprobs: List[float] = field(default_factory=list)
    finish_reason: List[str] = field(default_factory=list)


class AutoEngine:
    @classmethod
    def from_path(cls, model_path, **options):
        if "/" in model_path:
            return DataGenLLMEngine(model_path=model_path, **options)
        else:
            return OpenAI(model_name=model_path, **options)


class DataGenLLMEngine(LLMEngine):
    def generate(self, templated_contexts, top_p, temperature, max_tokens, **kwargs):
        results = Response()

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )

        # we explicitly add requests here, so that we can keep track of the request id
        for request_id, context in enumerate(templated_contexts):
            self.llm_engine.add_request(str(request_id), context, sampling_params)
        responses = self._run_engine(use_tqdm=True)

        # we need to do a bit of kung-fu here, given that there might be *less* or *more* responses
        # than the number of requests, e.g. if there are generation errors. Therefore, we build a dictionary
        # of responses based on the request id.
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


class OpenAI:
    def __init__(
        self,
        model_name="text-davinci-003",
    ):
        self._model_name = model_name
        self.operator = GPT.create_lm(model_name=self.model_name)

    @property
    def model_name(self):
        return self._model_name

    def generate(
        self, templated_contexts, max_tokens=1024, top_p=1.0, temperature=0.0, **kwargs
    ):
        results = Response()

        pbar = tqdm.tqdm(range(len(templated_contexts)))
        for context in range(0, len(templated_contexts), 20):
            batch = templated_contexts[context : context + 20]
            for _ in range(10):
                try:
                    output = self.operator.generate(batch, max_tokens=max_tokens)
                    break
                except Exception as e:
                    print(e)
                    print("retrying...")
                    time.sleep(2)
                    continue
            results.outputs += output
            results.finish_reason += ["stop"] * len(output)
            pbar.update(len(batch))
        return results
