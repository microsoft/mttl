from collections import defaultdict
from copy import deepcopy
import tqdm
import torch
import numpy as np
import pytorch_lightning as pl

from mttl.dataloader.ni_metrics import compute_metrics
from mttl.models.utils import transfer_batch_to_device
from mttl.evaluators.base import compute_task_aggregation


class MMLUEvaluator(object):
    def __init__(self, config, data_dir=None, max_input_length=None, device="cuda"):
        from mttl.datamodule.mmlu_data_module import MMLUDataModule

        self.config = deepcopy(config)
        self.device = device

        if data_dir is None:
            data_dir = config.data_dir

        if max_input_length is not None:
            self.config.max_input_length = max_input_length

        self.data_dir = data_dir
        self.datamodule = MMLUDataModule(
            self.config, data_dir=self.data_dir, for_generation=True
        )
        self.datamodule.setup("test")

    def evaluate(self, model, subsample=-1):
        was_train = model.training
        if was_train:
            model.eval()

        tokenizer = self.datamodule.tokenizer

        # DDP
        if hasattr(model, "module"):
            model = model.module

        all_predictions = []
        all_references = []
        task_names = []
        all_EM = []

        dataloader = self.datamodule.test_dataloader(subsample)
        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )
        for step, batch in pbar:
            task_name = batch.pop("task_names", None)
            sources_texts = batch.pop("sources_texts", None)
            labels_text = batch.pop("labels_texts", None)

            extra_kwargs = {}
            max_length = 5

            if self.config.model_family == "gpt":
                max_length += batch["input_ids"].shape[-1]
                extra_kwargs["pad_token_id"] = tokenizer.pad_token_id

            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, pl.LightningModule):
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

            # tokenizer merges space
            if tokenizer.mttl_merges_space:
                # space is in labels
                assert sources_texts[0][-1] != " "
                options = [" A", " B", " C", " D"]
            else:
                # space is in prompt
                assert sources_texts[0][-1] == " "
                options = ["A", "B", "C", "D"]

            logits = predictions.scores[0]
            probs = (
                torch.stack(
                    [
                        logits[
                            :,
                            tokenizer(options[0], add_special_tokens=False).input_ids[
                                -1
                            ],
                        ],
                        logits[
                            :,
                            tokenizer(options[1], add_special_tokens=False).input_ids[
                                -1
                            ],
                        ],
                        logits[
                            :,
                            tokenizer(options[2], add_special_tokens=False).input_ids[
                                -1
                            ],
                        ],
                        logits[
                            :,
                            tokenizer(options[3], add_special_tokens=False).input_ids[
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
            task_names += task_name

            eval_metrics = compute_metrics(
                predictions, [[r] for r in references], reduction="mean"
            )

            all_EM.append(eval_metrics["exact_match"])
            pbar.set_description(
                f"Task: {task_name[0] if task_name else None}, EM: {np.mean(all_EM):.4f}"
            )

        if was_train:
            model.train()

        eval_metrics = compute_metrics(
            all_predictions, [[r] for r in all_references], reduction="none"
        )
        return compute_task_aggregation(task_names, eval_metrics["exact_match"])
