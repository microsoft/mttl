import json
import os

try:
    from mix_eval.api.registry import register_model
    from mix_eval.evaluate import compute_metrics_p, eval, parse_args
    from mix_eval.models.base import ChatModel

    mixeval_available = True

except ImportError:
    mixeval_available = False
    register_model = lambda x: x


from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from mttl.datamodule.utils import get_tokenizer_with_args
from mttl.evaluators.base import GenerativeEvaluator
from mttl.models.expert_model import MultiExpertModel, MultiExpertModelConfig
from mttl.models.library.expert_library import ExpertLibrary


@dataclass
class MixEvalConfig:
    batch_size: int = 16
    model_name: str = "mix_eval_expert_adapter"
    benchmark: str = "mixeval_hard"
    data_path: str = None
    version: str = "2024-08-11"
    split: str = None
    output_dir: str = None
    verbose: bool = False


@register_model("mix_eval_expert_adapter")
class MultiExpertAdapter(ChatModel):
    def chunk_generate(
        self,
        inputs,
        model,
        tok,
        max_tokens: int,
        sliding_window: int = 128 * 1024,
        chunk_size: int = 2500,
        verbose: bool = False,
        chunked: bool = False,
        **kwargs,
    ):
        if chunked:
            raise ValueError("Chunked is not supported.")

        with torch.no_grad():
            input_ids = inputs.input_ids  # (b, n)
            attention_mask = inputs.attention_mask  # (b, n)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            **kwargs,
        )
        generated_ids = [
            output_ids[len(in_ids) :] for in_ids, output_ids in zip(input_ids, outputs)
        ]
        responses = tok.batch_decode(generated_ids, skip_special_tokens=True)
        return responses

    def __init__(self, args):
        self.model = args.model

        super().__init__(args)

        self.tokenizer = get_tokenizer_with_args(
            model_name=self.model.base_model_name_or_path,
            model_family="gpt",
            padding_side="left",
            truncation_side="left",
            for_generation=True,
        )

        self.SYSTEM_MESSAGE = {
            "role": "system",
            "content": "You are a helpful assistant.",
        }  # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}

        self.model_max_len = self.model.max_position_embeddings
        self.max_input_length_closeend = (
            min(self.model_max_len, self.max_input_length)
            - self.closeended_max_new_tokens
        )
        self.max_input_length_openend = (
            min(self.model_max_len, self.max_input_length)
            - self.openended_max_new_tokens
        )


class MixEvalEvaluator(GenerativeEvaluator):
    def __init__(
        self,
    ):
        super().__init__(config=MixEvalConfig())

        if not mixeval_available:
            raise ValueError(
                "MixEval is not installed. Please install it using `pip install mix-eval`."
            )

        self.download_data()

    def download_data(self):
        import shutil
        import subprocess

        import mix_eval

        repo_url = "https://github.com/Psycoy/MixEval.git"
        data_folder = "mix_eval/data"
        temp_dir = "/tmp/mixeval_repo"
        target_dir = os.path.join(os.path.dirname(mix_eval.__file__), "data")

        self.config.data_path = target_dir

        if os.path.exists(target_dir):
            return

        # Clone the repository
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True)

        # Copy the data folder to the target directory
        shutil.copytree(
            os.path.join(temp_dir, data_folder), target_dir, dirs_exist_ok=True
        )

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

    def evaluate(
        self,
        model,
        split=None,
        shuffle=False,
        subsample=-1,
        output_path=None,
        verbose=False,
        **kwargs,
    ):
        # inject model into config
        self.config.model = model
        self.config.verbose = verbose

        if split is not None:
            self.config.split = split

        if output_path is not None:
            self.config.output_dir = output_path
        else:
            raise ValueError("Output path is required for evaluation.")

        eval(self.config)
        compute_metrics_p(self.config)

        with open(os.path.join(self.config.output_dir, "score.json"), "r") as f:
            score = json.load(f)
        return score[self.config.model_name]["overall"]


if __name__ == "__main__":
    evaluator = MixEvalEvaluator()
    model = MultiExpertModel(
        MultiExpertModelConfig(base_model="microsoft/Phi-3-mini-4k-instruct"),
        device_map="cuda:0",
    )
    evaluator.evaluate(model, output_path="/tmp/mixeval/")
