# global interpreter
import asyncio
import numpy as np
import openai
import logging
import tiktoken
import os
from typing import List, Any
from dataclasses import dataclass
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


default_forward_lm = None
default_backward_lm = None
default_scoring_lm = None


@dataclass
class LogProbs:
    logp_targets: np.ndarray
    distribution: np.ndarray


@dataclass
class ScoreRequest:
    context: str
    target: str
    payload: Any = None


class GPT:
    AVAILABLE_MODELS = [
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "gpt-3.5-turbo",
        "gpt-35-turbo",
        "gpt-35-turbo-instruct",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-0613",
        "any",
    ]

    def __init__(self, model_name="text-davinci-003", **generation_options):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"model_name should be one of: {','.join(self.AVAILABLE_MODELS)}"
            )

        self.generation_options = generation_options
        self.engine = model_name
        self.api_type = os.environ.get("OPENAI_API_TYPE", "openai")

        if self.engine == "any":
            openai.api_base = "http://0.0.0.0:8081"
            openai.api_key = "any"
            openai.api_type = "openai"
            self.encoder = tiktoken.encoding_for_model("text-davinci-003")
        else:
            self.encoder = tiktoken.encoding_for_model(self.engine)

        openai.api_version = os.environ.get("OPENAI_API_VERSION")

    def encode(self, string):
        return self.encoder.encode(string)

    @property
    def has_log_probs(self):
        return not (
            "gpt-3.5" in self.engine
            or "gpt-4" in self.engine
            or "gpt-35" in self.engine
            or "gpt-4-0613" in self.engine
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )
    async def aget_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        if openai.api_type == "azure":
            try:
                response = await openai.ChatCompletion.acreate(
                    deployment_id=self.engine,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            except openai.InvalidRequestError as e:
                # Most likely a content filtering error from Azure.
                logging.warning(str(e))
                return str(e)
        else:
            response = await openai.ChatCompletion.acreate(
                model=self.engine,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

        if "content" not in response["choices"][0]["message"]:
            return ""

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
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )
    def get_chat_completion_response(self, prompt, **kwargs):
        """
        prompting chatgpt via openai api
        now batching only works for completion, not on chat
        """
        if openai.api_type == "azure":
            try:
                response = openai.ChatCompletion.create(
                    deployment_id=self.engine,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            except openai.InvalidRequestError as e:
                # Most likely a content filtering error from Azure.
                logging.warning(str(e))
                return str(e)
        else:
            response = openai.ChatCompletion.create(
                model=self.engine,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

        if "content" not in response["choices"][0]["message"]:
            return ""

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
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
    )
    def get_completion_response(
        self,
        prompt_batch,
        return_logprobs=False,
        raw_logprobs=False,
        top_logprobs=False,
        **kwargs,
    ):
        """
        prompting gpt-3 via openai api
        now batching only works for completion, not on chat
        """
        logging.debug(kwargs)

        try:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt_batch,
                logprobs=top_logprobs or 1,
                **kwargs,
            )
        except openai.InvalidRequestError as e:
            # Most likely a content filtering error from Azure.
            if "filtering" in str(e):
                logging.warning(str(e))
                # Process each element in the batch individually.
                response = {"choices": []}
                for prompt in prompt_batch:
                    try:
                        response["choices"].append(
                            openai.Completion.create(
                                engine=self.engine,
                                prompt=prompt,
                                logprobs=top_logprobs or 1,
                                **kwargs,
                            )["choices"][0]
                        )
                    except openai.InvalidRequestError as e:
                        response["choices"].append(
                            {
                                "text": str(e),
                                "logprobs": {
                                    "token_logprobs": [0],
                                    "top_logprobs": [{}],
                                    "tokens": {},
                                },
                            }
                        )
            else:
                raise e

        output = []
        nlls = []
        lengths = []
        for response in response["choices"]:
            output.append(response["text"].strip())
            if raw_logprobs:
                nlls.append(response["logprobs"]["token_logprobs"])
                lengths.append(response["logprobs"]["tokens"])
            elif top_logprobs:
                nlls.append(response["logprobs"]["top_logprobs"])
                lengths.append(response["logprobs"]["tokens"])
            else:
                if "token_logprobs" in response["logprobs"]:
                    nlls.append(sum(response["logprobs"]["token_logprobs"]))
                    lengths.append(len(response["logprobs"]["token_logprobs"]))
                else:
                    nlls.append(-np.inf)
                    lengths.append(1)

        if return_logprobs:
            output = list(zip(output, nlls, lengths))
        return output

    async def gather_chat_response(self, inputs, **generation_options):
        outputs = await asyncio.gather(
            *[
                self.aget_chat_completion_response(_input, **generation_options)
                for _input in inputs
            ]
        )
        return outputs

    def _mini_batch(self, inputs, batch_size=20):
        input_length = len(inputs)
        num_batches = input_length // batch_size + (
            1 if input_length % batch_size > 0 else 0
        )
        for i in range(num_batches):
            input_batch = inputs[batch_size * i : batch_size * (i + 1)]
            yield input_batch

    def generate(
        self,
        inputs,
        async_generation=True,
        batch_size=20,
        stream=False,
        progress=True,
        **kwargs,
    ):
        import tqdm

        if type(inputs) is not list:
            inputs = [inputs]

        kwargs.pop("output_space", None)
        generation_options = self.generation_options.copy()
        generation_options.update(**kwargs)

        progress_bar = tqdm.tqdm(inputs, disable=not progress)

        if self.engine in (
            "gpt-3.5-turbo",
            "gpt-35-turbo",
            "gpt-4",
            "gpt-4-32k",
            "gpt-4-0613",
            "any",
        ):
            if "return_logprobs" in generation_options:
                del generation_options["return_logprobs"]

            if async_generation is True:
                # async call api, devide to mini batches to avoid call rate limit
                outputs = []
                for input_batch in self._mini_batch(inputs, batch_size=batch_size):
                    outputs_batch = asyncio.run(
                        self.gather_chat_response(input_batch, **generation_options)
                    )
                    if stream:
                        yield from outputs_batch
                    else:
                        outputs = outputs + outputs_batch
                    progress_bar.update(len(input_batch))
            else:
                # call api one by one
                outputs = []
                for _input in inputs:
                    output = self.get_chat_completion_response(
                        _input, **generation_options
                    )
                    if stream:
                        yield output
                    else:
                        outputs.append(output)
                    progress_bar.update(1)
        else:
            # devide to mini batches (max batch size = 20 according to openai)
            outputs = []
            for input_batch in self._mini_batch(inputs, batch_size=batch_size):
                outputs_batch = self.get_completion_response(
                    input_batch, **generation_options
                )
                if stream:
                    yield from outputs_batch
                else:
                    outputs = outputs + outputs_batch
                progress_bar.update(len(input_batch))
        if not stream:
            return outputs

    def compute_cost(self, inputs):
        return np.sum(list([len(self.encoder(input)) for input in inputs]))

    def compute_log_p(
        self, requests: List[ScoreRequest], output_classes=None, agg="sum"
    ) -> LogProbs:
        """Compute log probability given a list of ScoreRequest objects.

        Args:
            requests: List of ScoreRequest objects.
            output_classes: If provided, compute the log probability of the target normalized across output classes.
            agg (str, optional): How to aggregate scores of different verbalizers for the same class.

        Returns:
            LogProbs
        """
        if not self.has_log_probs:
            raise ValueError(
                "This model ({}) does not support log probabilities.".format(
                    self.engine
                )
            )

        if output_classes is not None:
            return self._forward_logprobs_score_api_with_classes(
                [b.context for b in requests],
                [b.target for b in requests],
                output_classes,
                agg=agg,
            )
        return self._forward_logprobs_score_api(
            [b.context for b in requests],
            [b.target for b in requests],
        )

    def _forward_logprobs_score_api_with_classes(
        self, contexts, targets, output_classes, agg="max"
    ) -> LogProbs:
        eval_kwargs = {
            "temperature": 0.0,
            "max_tokens": 1,
            "echo": False,
            "return_logprobs": True,
            "raw_logprobs": False,
            "top_logprobs": 100,
        }

        unique_contexts = list(set(contexts))
        context_to_position = {context: i for i, context in enumerate(unique_contexts)}
        to_eval = [f"{context}\n" for context in unique_contexts]

        logging.debug("# Scoring requests = {}".format(len(contexts)))
        logging.debug("# Scoring unique requests = {}".format(len(unique_contexts)))
        eval_results = self.generate(
            to_eval,
            async_generation=True,
            **eval_kwargs,
        )

        top_logprobs = []
        for context in contexts:
            position = context_to_position[context]
            context_top_logprobs = eval_results[position][1][0]
            top_logprobs.append(dict(context_top_logprobs))

        output_logprobs = []
        output_distribs = []
        for context, target, context_top_logprobs in zip(
            contexts, targets, top_logprobs
        ):
            position = context_to_position[context]

            # make this fixed
            if context_top_logprobs:
                min_prob = np.exp(np.min(list(context_top_logprobs.values())))
            else:
                min_prob = 1e-6

            output_classes_scores = np.asarray([min_prob for _ in output_classes])
            # accumulate probability mass for each class verbalizer
            # the class verbalizer can be either " a" or "a" (with or without space)
            for i in range(len(output_classes)):
                verbalizers = output_classes.verbalizers(i)
                verbalizers.extend([f" {v}" for v in verbalizers])
                verbalizers = set(verbalizers)
                verbalizers_scores = [0.0]
                for verbalizer in verbalizers:
                    if verbalizer in context_top_logprobs:
                        prob_orig = np.exp(context_top_logprobs[verbalizer])
                    else:
                        prob_orig = min_prob
                    verbalizers_scores.append(prob_orig)
                if agg == "max":
                    output_classes_scores[i] += np.max(verbalizers_scores)
                else:
                    output_classes_scores[i] += np.sum(verbalizers_scores)

            output_class_index = [
                i
                for i, output_class in enumerate(output_classes)
                if target in output_class.split("|")
            ]
            assert (
                len(output_class_index) == 1
            ), "The target shouldn't appear in two output classes! {}".format(target)
            # accuracy here
            output_classes_scores = output_classes_scores / output_classes_scores.sum()
            output_logprobs.append(np.log(output_classes_scores[output_class_index[0]]))
            output_distribs.append(output_classes_scores)
        return LogProbs(np.asarray(output_logprobs), np.asarray(output_distribs))

    def _forward_logprobs_score_api(self, contexts, targets) -> LogProbs:
        logging.debug("# Scoring requests = {}".format(len(contexts)))

        eval_kwargs = {
            "temperature": 0,
            "max_tokens": 0,
            "echo": True,
            "return_logprobs": True,
            "raw_logprobs": True,
        }

        eval_batch = []
        for context, target in zip(contexts, targets):
            to_eval = f"{context}\n{target}"
            eval_batch.append(to_eval)

        # there might be doubles in the eval_batch, so we need to
        # only perform unique evals
        unique_keys = list(set(eval_batch))
        unique_keys_to_positions = {key: i for i, key in enumerate(unique_keys)}
        unique_eval_results = self.generate(
            unique_keys,
            async_generation=True,
            **eval_kwargs,
        )
        # get the results in the same order as the eval_batch
        eval_results = []
        for eval_key in eval_batch:
            eval_results.append(unique_eval_results[unique_keys_to_positions[eval_key]])
        # get the nll results
        log_probs = [eval_result[1] for eval_result in eval_results]

        # get the logprobs results
        output_logprobs = []
        context_logprobs = []

        for context, token_log_probs in zip(contexts, log_probs):
            num_tokens_prompt = len(self.encoder.encode(context))
            target_log_probs = token_log_probs[num_tokens_prompt:]
            context_log_probs = token_log_probs[1:num_tokens_prompt]

            if len(target_log_probs) == 0:
                output_logprobs.append("empty")
            else:
                output_logprobs.append(
                    sum(target_log_probs) / (len(target_log_probs) + 1e-5)
                )

            context_logprobs.append(
                sum(context_log_probs) / (len(context_log_probs) + 1e-5)
            )

        non_empty = [o for o in output_logprobs if o != "empty"]
        if len(non_empty) == 0:
            min = 0
        else:
            min = np.min(non_empty)
        output_logprobs = [o if o != "empty" else min for o in output_logprobs]
        return LogProbs(np.asarray(output_logprobs), np.asarray(context_logprobs))

    @classmethod
    def create_lm(cls, model_name="text-davinci-003", **generation_options):
        lm = cls(model_name, **generation_options)
        return lm
