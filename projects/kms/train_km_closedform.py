import copy
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from simple_utils import SimpleLogger, dcd_loss, do_evaluation
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from mttl.arguments import ExpertConfig
from mttl.datamodule.base import get_datamodule
from mttl.dist_utils import (
    get_device,
    get_local_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)
from mttl.logging import logger, setup_logging
from mttl.models.expert_model import ExpertModel, ExpertModelConfig, disable_modifiers
from mttl.models.get_optimizer import get_optimizer_and_scheduler
from mttl.models.modifiers.base import Modifier
from mttl.models.utils import transfer_batch_to_device
from mttl.utils import create_library, remote_login, upload_library

torch.set_float32_matmul_precision("high")

import torch


def closed_form_low_rank_solution(X, Y, k, lambda_reg=1e-5):
    """
    Finds W = AB where A and B minimize || Y - XAB ||_F^2 with rank k.

    :param X: Input tensor of shape (n, d)
    :param Y: Output tensor of shape (n, p)
    :param k: Desired rank (k < min(d, p))
    :param lambda_reg: Regularization parameter to ensure numerical stability
    :return: A (d, k), B (k, p)
    """
    n, d = X.shape
    _, p = Y.shape

    # Compute (X^T X) and its regularized inverse to prevent singularity
    XtX = X.T @ X  # Shape: (d, d)
    XtX_reg = XtX + lambda_reg * torch.eye(d, device=X.device)  # Regularization
    XtX_inv = torch.inverse(XtX_reg)  # Shape: (d, d)

    # Compute W_OLS (Ordinary Least Squares solution)
    W_OLS = XtX_inv @ X.T @ Y  # Shape: (d, p)

    # Project Y onto the column space of X
    Y_proj = X @ W_OLS  # Shape: (n, p)

    # Compute the SVD of the projected Y
    U_Y, S_Y, Vh_Y = torch.linalg.svd(Y_proj, full_matrices=False)
    V_Y = Vh_Y.T  # Convert V^H to V for real-valued data

    # Truncate to rank k
    U_k = U_Y[:, :k]  # Shape: (n, k)
    S_k = S_Y[:k]  # Shape: (k,)
    V_k = V_Y[:, :k]  # Shape: (p, k)

    # Compute the square roots of the singular values
    sqrt_S_k = torch.sqrt(S_k)  # Shape: (k,)

    # Compute A and B without forming large diagonal matrices
    A = XtX_inv @ X.T @ (U_k * sqrt_S_k)  # Shape: (d, k)
    B = sqrt_S_k.unsqueeze(1) * V_k.T  # Shape: (k, p)

    return A, B


# Example usage
n, d, p, k = 100, 50, 10, 5
X = torch.randn(n, d)
Y = torch.randn(n, p)

A, B = closed_form_low_rank_solution(X, Y, k)
W = A @ B  # Reconstructed weight matrix of shape (d, p)
print("Shape of W:", W.shape)  # Should be (d, p)
print("Rank of W:", torch.linalg.matrix_rank(W))  # Should be <= k

# Compute the reconstruction error
reconstruction_error = torch.norm(Y - X @ W, p="fro") ** 2
print("Reconstruction Error:", reconstruction_error.item())


@dataclass
class KMArguments(ExpertConfig):
    loss_function: str = "dcd"
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = "sordonia/narrativeqa_sanitized"


def train_km(training_args: KMArguments):
    seed_everything(training_args.seed, workers=True)

    # get directory of the current file
    setup_logging(training_args.output_dir)

    if is_main_process():
        training_args.save_config(training_args.output_dir)

    logger.info("Args: %s", training_args.to_json())

    remote_login(training_args.remote_token)

    model_config = ExpertModelConfig(
        base_model=args.model,
        task_name=args.finetune_task_name,
        expert_name=args.expert_name or args.finetune_task_name,
        modifier_config=args.modifier_config,
    )

    device = get_device()
    raw_model = model = ExpertModel(
        model_config,
        load_in_4bit=training_args.load_in_4bit,
        load_in_8bit=training_args.load_in_8bit,
        device_map=training_args.device_map,
        attn_implementation=training_args.attn_implementation,
    ).to(device)

    if is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[get_local_rank()])

    # load the NQA callback to monitor zero-shot performance
    from nqa_evaluator import NQAZeroShotEvaluator

    data_args = copy.deepcopy(training_args)
    data_args.dataset = training_args.nqa_dataset
    evaluator = NQAZeroShotEvaluator(data_args, generation_kwargs={})

    if training_args.loss_function == "dcd":
        loss_function = dcd_loss
    else:
        raise ValueError(f"Loss function {training_args.loss_function} not supported")

    datamodule = get_datamodule(training_args)
    val_loss, rougeL = do_evaluation(datamodule, model, loss_function, evaluator)

    # For every LoRA layer, let's set up a hook to fetch both the input and outputs of a forward pass
    # This is useful for computing the loss function
    def setup_hooks(model, info_container):
        def hook_fn(name, module, input, output):
            labels, nc_labels = info_container["labels"], info_container["nc_labels"]

            if module._enabled:  # store the input
                module.input += [input[0][nc_labels != -100].detach().cpu()]
            else:
                module.output += [output[labels != -100].detach().cpu()]

            assert abs(len(module.input) - len(module.output)) <= 1

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, Modifier):
                logger.info(f"Setting up hook for {name}")
                module.input = []
                module.output = []
                hook = module.register_forward_hook(partial(hook_fn, name))
                hooks.append(hook)

        return hooks

    def remove_hooks(hooks):
        for hook in hooks:
            hook.remove()

    info_container = {}
    hooks = setup_hooks(model, info_container)

    # without grad, do a pass over the training dataset
    with torch.no_grad():
        datamodule = get_datamodule(training_args)
        pbar = tqdm(datamodule.train_dataloader())
        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 16:
                break
            inputs = transfer_batch_to_device(batch, device)
            # document + small task prompt + task output (e.g. summary, or question and answer)
            input_ids = inputs["input_ids"]
            labels = inputs["labels"]
            attention_mask = inputs["attention_mask"]

            # small task prompt + task output (e.g. summary, or question and answer)
            nc_input_ids = inputs["nc_input_ids"]
            nc_labels = inputs["nc_labels"]
            nc_attention_mask = inputs["nc_attention_mask"]
            info_container["labels"] = labels
            info_container["nc_labels"] = nc_labels

            with disable_modifiers(raw_model):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            outputs = model(
                input_ids=nc_input_ids,
                attention_mask=nc_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Compute DCD loss between inputs and outputs
            # and check whether iterating over time helps

        # Now, we iterate over the layers, and solve closed form LoRA values
        for name, module in model.named_modules():
            if isinstance(module, Modifier):
                logger.info(f"Computing closed form LoRA values for {name}")
                # compute the closed form LoRA values
                inputs = torch.cat(module.input, dim=0).cuda().float()[::10]  # 20]
                outputs = torch.cat(module.output, dim=0).cuda().float()[::10]  # 20]
                A, B = closed_form_low_rank_solution(
                    inputs, outputs, module.config.lora_rank, lambda_reg=100
                )
                module.lora_a.data.copy_(A.data)
                module.lora_b.data.copy_(B.data)

        breakpoint()
        remove_hooks(hooks)
        val_loss, rougeL = do_evaluation(datamodule, model, loss_function, evaluator)
        xx = 1

    (optimizer, scheduler), trainable_param_names = get_optimizer_and_scheduler(
        model, training_args, num_train_examples=len(datamodule.train_dataset)
    )
    # compute number of trainable parameters
    num_trainable_params = sum(
        p.numel() for name, p in model.named_parameters() if p.requires_grad
    )
    logger.info(f"Number of trainable parameters: {num_trainable_params // 1e6:.2f}M")

    pbar = tqdm(
        total=len(datamodule.train_dataloader())
        * training_args.num_train_epochs
        // args.gradient_accumulation_steps
    )

    global_step = 0
    best_val = float("inf")
    met_logger = SimpleLogger(training_args.output_dir)

    val_loss, rougeL = do_evaluation(datamodule, model, loss_function, evaluator)
    met_logger.log_metrics({"val_loss": val_loss, "rougeL": rougeL}, step=global_step)

    for epoch in range(args.num_train_epochs):
        epoch_end = False

        iter_train = iter(datamodule.train_dataloader())
        while not epoch_end:
            loss_accum = 0.0
            model.train()
            optimizer.zero_grad()

            for step in range(args.gradient_accumulation_steps):
                try:
                    batch = next(iter_train)
                except StopIteration:
                    epoch_end = True
                    break

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    batch = transfer_batch_to_device(batch, device)

                loss = loss_function(model, batch)
                loss = loss / args.gradient_accumulation_steps
                loss_accum += loss.detach()
                loss.backward()

            if loss_accum:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()
                optimizer.step()
                torch.cuda.synchronize()  # wait for the GPU to finish work
                pbar.update(1)

                lr = optimizer.param_groups[0]["lr"]
                met_logger.log_metrics(
                    {"train_loss": loss_accum.item(), "grad_norm": norm, "lr": lr},
                    step=global_step,
                )
                logger.info(
                    f"Epoch {epoch}, Loss: {loss_accum.item():.5f}, Grad Norm: {norm:.5f}, LR: {lr:.6f}"
                )

            global_step += 1

        val_loss, rougeL = do_evaluation(datamodule, model, loss_function, evaluator)
        met_logger.log_metrics(
            {"val_loss": val_loss, "rougeL": rougeL}, step=global_step
        )

        if val_loss < best_val and is_main_process():
            best_val = val_loss
            raw_model.save_pretrained(training_args.output_dir + "/best_model")
            logger.info(f"Saving model to {training_args.output_dir}")


if __name__ == "__main__":
    args = KMArguments.parse()
    train_km(args)
