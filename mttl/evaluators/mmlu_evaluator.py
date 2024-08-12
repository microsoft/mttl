import copy
import hashlib
import os

import click
import numpy as np
import torch
import tqdm

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.evaluators.base import (
    GenerativeEvaluator,
    compute_task_aggregation,
    switch_to_eval_mode,
)
from mttl.vllm_engines.engines import LLMEngineMMLU, free_memory


def swap_model(model, state=None):
    """Swap the model to and from CPU."""
    if state is None:
        state = {}
    for n, p in model.named_parameters():
        if n in state:
            p.to(state[n])
        else:
            state[n] = p.device
            p.data = p.data.cpu()
    return state


class MMLUEvaluator(GenerativeEvaluator):
    def __init__(
        self,
        config=None,
        datamodule=None,
        max_input_length=None,
        use_vllm=False,
        generation_kwargs=None,
    ):
        from mttl.datamodule.mmlu_data_module import MMLUDataModule

        if datamodule is None:
            config = copy.deepcopy(config)

            if max_input_length is not None:
                config.max_input_length = max_input_length

            datamodule = MMLUDataModule(config, for_generation=True)

        generation_kwargs = generation_kwargs or {}
        generation_kwargs["max_new_tokens"] = 1

        super().__init__(
            datamodule, use_vllm=use_vllm, generation_kwargs=generation_kwargs
        )

    def eval_vllm(self, model, generation_config, subsample, shuffle):
        model_hash = hashlib.sha256()

        if hasattr(model, "hparams"):
            model_hash.update(f"{model.hparams}_{model.model.__class__}".encode())
        else:
            model_hash.update(f"{model.__class__}".encode())

        # move the model to CPU as VLLM loads its own version of the model
        state = swap_model(model)

        vllm_model = LLMEngineMMLU(
            model,
            temp_path=f"{os.environ.get('MTTL_TEMP', '/tmp/merged')}/{model_hash.hexdigest()}/",
        )

        if self.split == "test":
            dataloader = self.datamodule.test_dataloader(subsample, shuffle)
        else:
            dataloader = self.datamodule.val_dataloader()

        all_predictions, all_references, all_task_names = vllm_model.eval(
            dataloader, generation_config, self.tokenizer
        )

        free_memory()
        del vllm_model

        # move the model back to GPU
        swap_model(model, state)

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        return compute_task_aggregation(all_task_names, eval_metrics["exact_match"])

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="test",
        subsample=-1,
        num_batches=None,
        shuffle=False,
        output_path=None,
        **kwargs,
    ):
        if self.use_vllm:
            metrics = self.eval_vllm(
                model,
                generation_config=model.generation_config,
                subsample=subsample,
                shuffle=shuffle,
            )
        else:
            all_predictions = []
            all_references = []
            all_task_names = []
            all_EM = []

            dataloader = self.get_dataloader(split, subsample, shuffle)

            pbar = tqdm.tqdm(
                enumerate(dataloader),
                total=len(dataloader),
            )
            for num_batch, batch in pbar:
                if num_batches and num_batch >= num_batches:
                    break

                task_names = batch.get("task_names", None)
                labels_text = batch.pop("labels_texts", None)

                predictions = self.generate_for_batch(model, batch)
                logits = predictions.scores[0]

                probs = (
                    torch.stack(
                        [
                            logits[
                                :,
                                self.tokenizer(
                                    " A", add_special_tokens=False
                                ).input_ids[-1],
                            ],
                            logits[
                                :,
                                self.tokenizer(
                                    " B", add_special_tokens=False
                                ).input_ids[-1],
                            ],
                            logits[
                                :,
                                self.tokenizer(
                                    " C", add_special_tokens=False
                                ).input_ids[-1],
                            ],
                            logits[
                                :,
                                self.tokenizer(
                                    " D", add_special_tokens=False
                                ).input_ids[-1],
                            ],
                        ],
                        dim=-1,
                    )
                    .max(dim=-1)[1]
                    .detach()
                    .cpu()
                    .numpy()
                )

                predictions = [{0: "A", 1: "B", 2: "C", 3: "D"}[p] for p in probs]
                references = labels_text

                all_predictions += predictions
                all_references += references
                all_task_names += task_names

                eval_metrics = compute_metrics(
                    predictions, [[r] for r in references], reduction="none"
                )

                all_EM.extend(eval_metrics["exact_match"])
                pbar.set_description(
                    f"Task: {task_names[0] if task_names else None}, EM: {np.mean(all_EM):.4f}"
                )

            eval_metrics = compute_metrics(
                all_predictions, [[r] for r in all_references], reduction="none"
            )
            metrics = compute_task_aggregation(
                all_task_names, eval_metrics["exact_match"]
            )

        self.save_metrics(metrics, output_path)
        return metrics["all"]["mean"]


class MMLUEvaluatorFast(MMLUEvaluator):
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs, num_batches=400, shuffle=True)


@click.command()
@click.argument("hf_model")
@click.option("--task_name", default=None)
def evaluate_mmlu(hf_model, task_name=None):
    from mttl.datamodule.mmlu_data_module import MMLUDataConfig
    from mttl.models.utils import model_loader_helper

    model = model_loader_helper(hf_model)
    config = MMLUDataConfig(
        dataset="mmlu",
        model=hf_model,
        predict_batch_size=4,
        max_input_length=model.config.max_position_embeddings,
        model_family="gpt",
        finetune_task_name=task_name,
    )

    MMLUEvaluator(config).evaluate(model, shuffle=True)


if __name__ == "__main__":
    evaluate_mmlu()
