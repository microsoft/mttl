import numpy as np
import torch
import tqdm

from mttl.evaluators.base import Evaluator, switch_to_eval_mode
from mttl.logging import logger
from mttl.models.utils import compute_loglike_loss


class LossEvaluator(Evaluator):
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
        from mttl.models.expert_model import ExpertModel
        from mttl.models.lightning.expert_module import ExpertModule
        from mttl.models.utils import transfer_batch_to_device

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
        losses = []

        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            batch = transfer_batch_to_device(batch, device)

            with torch.no_grad():
                if isinstance(model, ExpertModule) or isinstance(model, ExpertModel):
                    loss = model.forward(**batch).loss
                else:
                    loss = model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).loss
            losses.append(loss.item())
            pbar.set_description("Loss: {:.4f}".format(np.mean(losses)))

        self.save_metrics({"loss": np.mean(losses)}, output_path)
        return np.mean(losses)
