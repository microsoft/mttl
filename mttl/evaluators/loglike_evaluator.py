import numpy as np
import torch
import tqdm

from mttl.evaluators.base import Evaluator, switch_to_eval_mode
from mttl.logging import logger
from mttl.models.expert_model_hf_base import BaseExpertModel
from mttl.models.utils import EfficientCheckpointModule, transfer_batch_to_device


def compute_loglike_loss(logits, labels, reduction="none"):
    # calculate loss, could also be done inside of the model
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # reshape back
    if reduction == "none":
        loss = loss.view((bs, -1))
        # mean only non-zero
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss
    return loss


class LogLikeEvaluator(Evaluator):
    def __init__(self, datamodule, **kwargs):
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
        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        if self.use_vllm:
            return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        all_losses = []
        all_accuracies = []
        all_predictions = []

        device = next(model.parameters()).device

        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            batch_size = len(batch["labels_index"])
            num_options = batch["num_options"]
            labels_texts = batch["labels_texts"]
            sources_texts = batch["sources_texts"]

            batch = transfer_batch_to_device(batch, device)

            with torch.no_grad():
                if isinstance(model, EfficientCheckpointModule):
                    # lightning module
                    loss_per_option = model.forward(batch, reduction="none")
                elif isinstance(model, BaseExpertModel):
                    # standard no lightning evaluation
                    loss_per_option, _ = model.forward(**batch, reduction="none")
                else:
                    logits = model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                    loss_per_option = compute_loglike_loss(
                        logits, batch["labels"], reduction="none"
                    )

                loss_per_option = loss_per_option.cpu().numpy()
                loss_per_example = [
                    loss_per_option[
                        int(np.sum(num_options[:i])) : int(np.sum(num_options[: i + 1]))
                    ]
                    for i in range(batch_size)
                ]
                predictions = [
                    np.argmin(option_loss) for option_loss in loss_per_example
                ]

                all_predictions.extend(predictions)
                all_losses.extend(loss_per_option.tolist())

                if "labels_index" in batch:
                    all_accuracies.extend(
                        (
                            np.array(predictions) == np.array(batch["labels_index"])
                        ).tolist()
                    )

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[batch["labels_index"][0]])
                logger.info("Prediction:\n%s", labels_texts[predictions[0]])

            if all_accuracies:
                pbar.set_description("Accuracy: {:.4f}".format(np.mean(all_accuracies)))

        metrics = {
            "loss": float(np.mean(all_losses)),
            "loglike": -float(np.mean(all_losses)),
            "predictions": all_predictions,
            "accuracy": float(np.mean(all_accuracies)) if all_accuracies else None,
        }

        self.save_metrics(metrics, output_path)
        return metrics["accuracy"]
