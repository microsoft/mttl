from typing import List
import numpy as np
import torch
import gc
import tqdm
import time
import os
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from mttl.utils import logger
from mttl.models.adapters import MergableAdapter


def save_merged_model(model, hf_path="/tmp/merged"):
    if os.path.exists(hf_path):
        return hf_path

    merged = []
    for name, module in model.model.named_modules():
        for c_name, child in module.named_children():
            if isinstance(child, MergableAdapter):
                child.merge_with_layer()
                setattr(
                    module,
                    c_name,
                    child.layer,
                )
                merged.append(name)

    logger.info("Merged layers: %s" % merged)
    logger.info("Saving merged model to: %s" % hf_path)

    model.model.save_pretrained(hf_path, save_full_model=True)
    logger.info("Saving tokenizer to: %s" % hf_path)
    model.tokenizer.save_pretrained(hf_path)
    return hf_path


def free_memory():
    from ray import shutdown

    gc.collect()
    torch.cuda.empty_cache()
    destroy_model_parallel()
    shutdown()
    time.sleep(3)


class LLMEngineMMLU(LLM):
    def __init__(self, model, temp_path="/tmp/merged", **options):
        # merge adapters -- if needed --
        path = save_merged_model(model, hf_path=temp_path)
        self.path = path
        options["model"] = path

        LLM.__init__(
            self,
            gpu_memory_utilization=0.85,
            disable_log_stats=False,
            swap_space=10,
            **options,
        )

    def eval(
        self,
        dataloader: DataLoader,
        generation_config,
        tokenizer,
        **kwargs,
    ):
        all_references = {}
        all_task_names = {}

        logprobs_for = 20
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=5,
            logprobs=logprobs_for,
        )
        target_to_id = {
            "A": tokenizer("A", add_special_tokens=False).input_ids[-1],
            "B": tokenizer("B", add_special_tokens=False).input_ids[-1],
            "C": tokenizer("C", add_special_tokens=False).input_ids[-1],
            "D": tokenizer("D", add_special_tokens=False).input_ids[-1],
        }

        # we explicitly add requests here, so that we can keep track of the request id
        for request_id, batch in enumerate(
            tqdm.tqdm(dataloader, total=len(dataloader))
        ):
            for context, label, task_name in zip(
                batch["sources_texts"], batch["labels_texts"], batch["task_names"]
            ):
                self.llm_engine.add_request(str(request_id), context, sampling_params)
                all_references[str(request_id)] = label
                all_task_names[str(request_id)] = task_name
        responses = self._run_engine(use_tqdm=True)
        responses_dict = {r.request_id: r for r in responses}
        # for each sample, for each token a list of logprobs of the logprobs_for most likely tokens

        _all_predictions = []
        _all_references = []
        _all_task_names = []
        for request_id, response in responses_dict.items():
            _all_task_names.append(all_task_names[request_id])
            _all_references.append(all_references[request_id])

            logprobs_first_tok = response.outputs[0].logprobs[0]
            max_logprob = -torch.inf
            _all_predictions.append(np.random.choice(list(target_to_id.keys())))
            for prediction, tok_id in target_to_id.items():
                if (
                    tok_id in logprobs_first_tok
                    and logprobs_first_tok[tok_id] > max_logprob
                ):
                    max_logprob = logprobs_first_tok[tok_id]
                    _all_predictions[-1] = prediction
        del self.llm_engine
        return _all_predictions, _all_references, _all_task_names

    def free_memory(self):
        if os.path.exists(self.path):
            # remvoe directory
            os.system("rm -rf %s" % self.path)
        free_memory()
