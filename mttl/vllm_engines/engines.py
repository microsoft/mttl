import copy
import gc
import os
import time

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from mttl.models.modifiers.base import MergeableAdapter
from mttl.utils import logger

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
except ImportError:
    LLM = object
    logger.warning("VLLM is not installed. Please install it to use LLMEngine.")


def save_merged_model(model, model_path, hf_path="/tmp/merged"):
    """
    This prioritizes model_path: if both model and model_path are given, it loads model from model_path.
    For the model, this assumes the model is on CPU. Creates a copy of the model and merges all adapters
    if needed. Then saves the model to the given path.
    """

    if model_path:
        # TODO: REMOVE this
        from mttl.models.expert_model import ExpertModel

        logger.info("Model path is given. Loading model from: %s" % model_path)

        model = ExpertModel.from_pretrained(
            model_path,
            load_in_8bit=False,
            device_map={"": "cpu"},
        )
        hf_path = os.path.join(hf_path, model_path.replace("/", "_"))

    for _, p in model.named_parameters():
        if p.device != torch.device("cpu"):
            raise ValueError("Model must be on CPU to merge adapters.")

    if not hasattr(model, "model"):
        raise ValueError("Model must have a `model` attribute, a HuggingFace model.")

    # if path already exists, we don't need to do anything
    if os.path.exists(hf_path):
        return hf_path
    else:
        os.makedirs(hf_path)

    merged = []
    model_copy = copy.deepcopy(model.model)

    for name, module in model_copy.named_modules():
        for c_name, child in module.named_children():
            if isinstance(child, MergeableAdapter):
                child.merge_with_layer()
                setattr(
                    module,
                    c_name,
                    child.layer,
                )
                merged.append(name)

    logger.info("Merged layers: %s" % merged)
    logger.info("Saving merged model to: %s" % hf_path)

    model_copy.save_pretrained(hf_path, save_full_model=True)

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


class LLMEngine(LLM):
    def __init__(self, model=None, model_path=None, temp_path="/tmp/merged", **options):
        assert (
            model is not None or model_path is not None
        ), "Either model or model_path must be given."
        # merge adapters -- if needed --
        path = save_merged_model(model, model_path, hf_path=temp_path)
        options["model"] = path

        LLM.__init__(
            self,
            **options,
        )

        if os.path.exists(path):
            # remvoe directory
            os.system("rm -rf %s" % path)

    @property
    def model_name(self):
        return self.llm_engine.model_config.model


class LLMEngineRouge(LLMEngine):
    def eval(
        self,
        dataloader: DataLoader,
        generation_config,
        max_tokens,
        **kwargs,
    ):
        raise NotImplementedError("This is not finished yet.")
        all_references = {}
        all_task_names = {}
        sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=max_tokens,
        )

        # we explicitly add requests here, so that we can keep track of the request id
        for request_id, batch in enumerate(
            tqdm.tqdm(dataloader, total=len(dataloader))
        ):
            for context, label in zip(batch["sources_texts"], batch["labels_texts"]):
                self.llm_engine.add_request(str(request_id), context, sampling_params)
                all_references[str(request_id)] = label

        responses = self._run_engine(use_tqdm=True)
        responses_dict = {r.request_id: r for r in responses}

        _all_predictions = []
        _all_references = []
        for request_id, response in responses_dict.items():
            _all_references.append(all_references[request_id])

        return _all_predictions, _all_references


class LLMEngineMMLU(LLMEngine):
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
            "A": tokenizer(" A", add_special_tokens=False).input_ids[-1],
            "B": tokenizer(" B", add_special_tokens=False).input_ids[-1],
            "C": tokenizer(" C", add_special_tokens=False).input_ids[-1],
            "D": tokenizer(" D", add_special_tokens=False).input_ids[-1],
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
