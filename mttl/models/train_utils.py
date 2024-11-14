import os

import torch
from tqdm.auto import tqdm

from mttl.datamodule.base import DataModule
from mttl.logging import logger
from mttl.models.base_model import WEIGHTS_NAME, BaseExpertModel
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.utils import transfer_batch_to_device


@torch.no_grad()
def evaluate_model(dataloader, model):
    """Evaluation loop."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        with torch.autocast(
            device_type=model.device.type,
            dtype=model.dtype,
        ):
            batch = transfer_batch_to_device(batch, model.device)
            output = model.forward(**batch)
            total_loss += output.loss.item()
            total_samples += 1
    return total_loss / total_samples


def train_model(
    args: "TrainingArguments",
    model: BaseExpertModel,
    datamodule: DataModule,
    do_test=False,
) -> BaseExpertModel:
    """Mini-training loop."""
    import copy

    args = copy.deepcopy(args)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    (optimizer, scheduler), _ = get_optimizer_and_scheduler(
        model, args, num_train_examples=len(datamodule.train_dataset)
    )
    dataloader = datamodule.train_dataloader()
    num_train_steps = len(dataloader)
    iter_train = iter(dataloader)

    if args.eval_every_n_epoch != -1:
        args.eval_every = num_train_steps * args.eval_every_n_epoch

    bar = tqdm(range(args.total_steps))
    best_val_loss = float("inf")
    running_loss = 0.0

    for step in bar:
        loss_accum = 0.0
        model.train()
        optimizer.zero_grad()

        for micro_step in range(args.gradient_accumulation_steps):
            try:
                batch = next(iter_train)
            except StopIteration:
                iter_train = iter(dataloader)
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
                f"Step {step + 1}/{args.total_steps},"
                f" Loss: {running_loss / (step + 1):.4f},"
                f" Lr: {scheduler.get_last_lr()[0]:.4f},"
                f" Val: {best_val_loss:.4f}"
            )

        # eval and save best model
        if (
            args.eval_every > 0
            and step % args.eval_every == 0
            and datamodule.dev_dataset
        ):
            val_loss = evaluate_model(datamodule.val_dataloader(), model)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.output_dir:
                    model.save_pretrained(args.output_dir + "/best_model")
            running_loss = 0.0

    # reload best model
    if args.output_dir and os.path.exists(
        args.output_dir + f"/best_model/{WEIGHTS_NAME}"
    ):
        logger.info("Reloading best model!")

        model.load_state_dict(
            torch.load(
                args.output_dir + f"/best_model/{WEIGHTS_NAME}", weights_only=True
            ),
            strict=False,
        )

    # do test evaluation
    if do_test and datamodule.test_dataset:
        test_loss = evaluate_model(datamodule.test_dataloader(), model)
        logger.info(f"Test loss: {test_loss:.4f}")

    return model
