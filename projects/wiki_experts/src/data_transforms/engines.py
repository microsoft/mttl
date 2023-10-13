from typing import List
import numpy as np
import torch
import gc
import tqdm
import time
import os
from dataclasses import dataclass, field

from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from mttl.models.adapters import LoRA
from mttl.models.openai import GPT
from src.data_transforms.utils import INVALID_RESPONSE


@dataclass
class Response:
    contexts: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    cumulative_logprobs: List[float] = field(default_factory=list)
    finish_reason: List[str] = field(default_factory=list)


def free_memory():
    from ray import shutdown

    gc.collect()
    torch.cuda.empty_cache()
    destroy_model_parallel()
    shutdown()
    time.sleep(3)


def save_merged_model(mttl_ckpt_path, hf_path="/tmp/merged"):
    from projects.wiki_experts.src.expert_trainer import ExpertTrainer
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


class AutoEngine:
    @classmethod
    def from_path(cls, model_path, **options):
        if "/" in model_path:
            return LLMEngine(model_path, **options)
        else:
            return OpenAI(model_path, **options)


class LLMEngine(LLM):
    def __init__(self, model, temp_path="/tmp/merged", **options):
        # merge adapters -- if needed --
        path = save_merged_model(model, hf_path=temp_path)
        options["model"] = path

        LLM.__init__(self, **options)

    @property
    def model_name(self):
        return self.llm_engine.model_config.model

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
            output = self.operator.generate(batch, max_tokens=max_tokens)
            results.outputs += output
            results.finish_reason += ["stop"] * len(output)
            pbar.update(len(batch))
        return results
