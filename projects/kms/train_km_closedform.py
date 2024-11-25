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
    # val_loss, rougeL = do_evaluation(datamodule, model, loss_function, evaluator)

    # For every LoRA layer, let's set up a hook to fetch both the input and outputs of a forward pass
    # This is useful for computing the loss function
    def setup_hooks(model, info_container):
        def hook_fn(name, module, input, output):
            labels, nc_labels = info_container["labels"], info_container["nc_labels"]
            use_context = info_container["use_context"]

            if use_context:  # store the input
                module.input += [input[0][labels != -100].detach().cpu()]
                module.output += [output[labels != -100].detach().cpu()]
            else:
                module.nc_input += [input[0][nc_labels != -100].detach().cpu()]
                module.nc_output += [output[nc_labels != -100].detach().cpu()]

            assert abs(len(module.nc_input) - len(module.input)) <= 1

        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, Modifier):
                logger.info(f"Setting up hook for {name}")
                module.nc_input = []
                module.nc_output = []
                module.input = []
                module.output = []
                hook = module.register_forward_hook(partial(hook_fn, name))
                hooks.append(hook)

        return hooks

    def remove_hooks(hooks):
        for hook in hooks:
            hook.remove()

    info_container = {}
    for it in range(10):
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

                # No context forward pass
                info_container["use_context"] = True
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Context forward pass
                info_container["use_context"] = False
                outputs = model(
                    input_ids=nc_input_ids,
                    attention_mask=nc_attention_mask,
                )

            # Now, we iterate over the layers, and solve closed form LoRA values
            for name, module in model.named_modules():
                if isinstance(module, Modifier):
                    logger.info(f"Computing closed form LoRA values for {name}")
                    # compute the closed form LoRA values
                    # As, Bs = [], []
                    # for input, output in zip(module.input, module.output):
                    #     A, B = closed_form_low_rank_solution(
                    #         input[::5].float(), output[::5].float(), module.config.lora_rank, lambda_reg=100
                    #     )
                    #     As.append(A)
                    #     Bs.append(B)
                    #
                    # A = torch.stack(As).mean(dim=0)
                    # B = torch.stack(Bs).mean(dim=0)
                    input = torch.cat(module.input, dim=0)
                    nc_input = torch.cat(module.nc_input, dim=0)
                    output = torch.cat(module.output, dim=0)
                    nc_output = torch.cat(module.nc_output, dim=0)

                    print(f"diff input: {torch.norm(input - nc_input)}")
                    print(f"diff output: {torch.norm(output - nc_output)}")

                    Y = (output - nc_output).float().cuda()
                    X = (nc_input).float().cuda()

                    A, B = closed_form_low_rank_solution(
                        X[::20],
                        Y[::20],
                        module.config.lora_rank,
                        lambda_reg=100,
                    )

                    # breakpoint()
                    # pred y
                    pred_y = X @ A @ B
                    print(f"pred_y: {torch.norm(pred_y - Y)}")

                    module.lora_a.data.copy_(A.data)
                    module.lora_b.data.copy_(B.data)
                    # '''

            remove_hooks(hooks)
            val_loss, rougeL = do_evaluation(
                datamodule, model, loss_function, evaluator
            )
            xx = 1


if __name__ == "__main__":
    args = KMArguments.parse(raise_error=False)
    train_km(args)
