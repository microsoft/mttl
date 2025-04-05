import time

import numpy as np
import torch
from tqdm.auto import tqdm

from mttl.dist_utils import distributed_mean, is_main_process
from mttl.evaluators.base import Evaluator, switch_to_eval_mode
from mttl.logging import logger
from mttl.models.utils import compute_loglike_loss
from mttl.utils import disable_cache, get_raw_model


class LogLikeEvaluator(Evaluator):
    def __init__(self, datamodule, **kwargs):
        self.length_normalization = kwargs.pop("length_normalization", True)

        super().__init__(datamodule=datamodule, **kwargs)

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="val",
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
        output_path=None,
    ):
        from mttl.models.expert_model import BaseExpertModel
        from mttl.models.lightning.base_module import LightningEfficientCheckpoint
        from mttl.models.utils import transfer_batch_to_device

        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        if self.use_vllm:
            return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not is_main_process(),
            leave=False,
            position=1,
        )

        all_losses = []
        all_accuracies = []
        all_predictions = []
        time_per_request = []
        tokens_per_request = []
        device = next(model.parameters()).device

        raw_model = get_raw_model(model)

        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            labels_index = batch.pop("labels_index", None)
            num_options = batch.pop("num_options")
            labels_texts = batch.pop("labels_texts")
            sources_texts = batch.pop("sources_texts")
            batch_size = len(labels_index)

            batch = transfer_batch_to_device(batch, device)

            with torch.no_grad():
                with disable_cache(model, apply="phi3" in str(type(model))):
                    start = time.time()
                    if isinstance(
                        raw_model, LightningEfficientCheckpoint
                    ) or isinstance(raw_model, BaseExpertModel):
                        logits = model.forward(**batch).logits
                    else:
                        logits = model.forward(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        ).logits
                    end = time.time()
                    loss_per_option = compute_loglike_loss(
                        logits,
                        batch["labels"],
                        reduction="none",
                        normalize_length=self.length_normalization,
                    )
                    if loss_per_option.dtype in [torch.bfloat16, torch.float16]:
                        loss_per_option = loss_per_option.float()

                    loss_per_option = loss_per_option.cpu().numpy()
                    loss_per_example = [
                        loss_per_option[
                            int(np.sum(num_options[:i])) : int(
                                np.sum(num_options[: i + 1])
                            )
                        ]
                        for i in range(batch_size)
                    ]

                    predictions = [
                        np.argmin(option_loss) for option_loss in loss_per_example
                    ]

                    # get number of tokens in prompt
                    label_pad_id = self.datamodule.collate_fn.label_pad_token_id
                    num_tokens_in_prompt = (
                        (
                            batch["attention_mask"].sum(1)
                            - batch["labels"].ne(label_pad_id).sum(1)
                        )
                        .cpu()
                        .tolist()
                    )

                    num_tokens_in_prompt = [
                        num_tokens_in_prompt[
                            int(np.sum(num_options[:i])) : int(
                                np.sum(num_options[: i + 1])
                            )
                        ][0]
                        for i in range(batch_size)
                    ]

                    all_predictions.extend(predictions)
                    all_losses.extend(loss_per_option.tolist())
                    time_per_request.append((end - start) / batch_size)
                    tokens_per_request.extend(num_tokens_in_prompt)

                    if labels_index is not None:
                        all_accuracies.extend(
                            (np.array(predictions) == np.array(labels_index)).tolist()
                        )

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[labels_index[0]])
                logger.info("Prediction:\n%s", labels_texts[predictions[0]])

            if all_accuracies:
                pbar.set_description(
                    "Accuracy: {:.4f} \t Loss {:.4f} \t Time: {:.4f}".format(
                        np.mean(all_accuracies),
                        np.mean(all_losses),
                        np.mean(time_per_request),
                    )
                )

            del batch
            del logits
            del loss_per_option
            torch.cuda.empty_cache()

        loss = distributed_mean(all_losses, device)
        all_accuracies = (
            distributed_mean(all_accuracies, device) if all_accuracies else None
        )
        time_per_request = distributed_mean(time_per_request, device)
        tokens_per_request = distributed_mean(tokens_per_request, device)

        metrics = {
            "loss": loss,
            "loglike": -loss,
            "predictions": all_predictions,
            "accuracy": all_accuracies,
            "time_per_request": time_per_request,
            "tokens_per_request": tokens_per_request,
        }

        if output_path:
            self.save_metrics(metrics, output_path)
        return metrics["accuracy"]
