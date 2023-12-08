import os
import click
import tqdm
import torch
import hashlib
import numpy as np
import pytorch_lightning as pl

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.models.utils import transfer_batch_to_device
from mttl.evaluators.base import (
    compute_task_aggregation,
    Evaluator,
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


class MMLUEvaluator(Evaluator):
    def __init__(
        self, config=None, max_input_length=None, device="cuda", use_vllm=False
    ):
        super().__init__(config=config, device=device, use_vllm=use_vllm)

        from mttl.datamodule.mmlu_data_module import MMLUDataModule

        if max_input_length is not None:
            self.config.max_input_length = max_input_length

        self.datamodule = MMLUDataModule(self.config, for_generation=True)

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
        self, model, split="test", subsample=-1, shuffle=False, dataloader=None
    ):
        if self.use_vllm:
            return self.eval_vllm(
                model,
                generation_config=model.generation_config,
                subsample=subsample,
                shuffle=shuffle,
            )

        # DDP
        if hasattr(model, "module"):
            model = model.module
        model.to(self.device)

        all_predictions = []
        all_references = []
        all_task_names = []
        all_EM = []

        dataloader = self.get_dataloader(split, subsample, shuffle)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        for _, batch in pbar:
            task_names = batch.get("task_names", None)
            batch.pop("sources_texts", None)
            labels_text = batch.pop("labels_texts", None)
            extra_kwargs = {}
            max_length = 5

            if self.config.model_family == "gpt":
                max_length += batch["input_ids"].shape[-1]
                extra_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, pl.LightningModule) or hasattr(model, "hparams"):
                    predictions = model.generate(
                        batch,
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )
                else:
                    predictions = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=max_length,
                        generation_config=model.generation_config,
                        return_dict_in_generate=True,
                        output_scores=True,
                        **extra_kwargs,
                    )

            logits = predictions.scores[0]
            probs = (
                torch.stack(
                    [
                        logits[
                            :,
                            self.tokenizer(" A", add_special_tokens=False).input_ids[
                                -1
                            ],
                        ],
                        logits[
                            :,
                            self.tokenizer(" B", add_special_tokens=False).input_ids[
                                -1
                            ],
                        ],
                        logits[
                            :,
                            self.tokenizer(" C", add_special_tokens=False).input_ids[
                                -1
                            ],
                        ],
                        logits[
                            :,
                            self.tokenizer(" D", add_special_tokens=False).input_ids[
                                -1
                            ],
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
        return compute_task_aggregation(all_task_names, eval_metrics["exact_match"])


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
        predict_batch_size=16,
        max_input_length=model.config.max_position_embeddings,
        model_family="gpt",
        finetune_task_name=task_name,
    )

    MMLUEvaluator(config).evaluate(model, shuffle=True)


if __name__ == "__main__":
    evaluate_mmlu()
