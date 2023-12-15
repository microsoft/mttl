from mttl.models.utils import transfer_batch_to_device
from mttl.evaluators.base import Evaluator
from mttl.evaluators.rouge_evaluator import RougeEvaluator
import tqdm
import torch
from projects.wiki_experts.src.expert_trainer import ExpertTrainer
import numpy as np


class LossEvaluator(RougeEvaluator):
    def get_loss(
        self, model, split="val", subsample=-1, num_batches=None, verbose=True
    ):
        all_loss = []

        dataloader = self.get_dataloader(
            split="test", subsample=subsample, shuffle=False
        )

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        for _, batch in pbar:
            batch = transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                if isinstance(model, ExpertTrainer):
                    loss = model.get_loss_for_all(batch, 0)
                    all_loss.extend(loss.cpu().numpy().astype(np.float64))
            pbar.set_description(f"Loss Score: {np.mean(all_loss):.4f}")

            if num_batches is not None and len(all_loss) >= num_batches:
                break
        return np.mean(all_loss)
