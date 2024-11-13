import torch
from tqdm.auto import tqdm

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import DataModule
from mttl.models.base_model import BaseExpertModel
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device


def train_model(
    args: ExpertConfig, model: BaseExpertModel, datamodule: DataModule
) -> BaseExpertModel:
    """Mini-training loop."""
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    (optimizer, scheduler), _ = get_optimizer_and_scheduler(
        model, args, num_train_examples=len(datamodule.train_dataset)
    )
    iter_train = iter(datamodule.train_dataloader())

    bar = tqdm(range(args.total_steps))
    running_loss = 0.0
    for step in bar:
        loss_accum = 0.0
        model.train()
        optimizer.zero_grad()

        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(datamodule.train_dataloader())
                batch = next(iter_train)

            with torch.autocast(
                device_type=model.device.type,
                dtype=model.dtype,
            ):
                batch = transfer_batch_to_device(batch, model.device)
                loss = model.forward(**batch).loss
                loss = loss / args.gradient_accumulation_steps
                loss_accum += loss.detach()
                loss.backward()

        if loss_accum:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            running_loss += loss_accum.item()
            optimizer.step()
            scheduler.step()
            if model.device.type == "cuda":
                torch.cuda.synchronize()

            bar.set_description_str(
                f"Step {step + 1}/{args.total_steps}, Loss: {running_loss / (step + 1):.4f}, Lr: {scheduler.get_last_lr()[0]:.4f}"
            )
    return model
