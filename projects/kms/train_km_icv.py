import copy
from dataclasses import dataclass

import torch
from icv_utils import PCA

# register this datamodule!
from km_datamodule import KMDatasetModule
from lightning_fabric import seed_everything
from simple_utils import dcd_loss, do_evaluation
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
from mttl.models.expert_model import ExpertModel, ExpertModelConfig
from mttl.models.modifiers.base import Modifier
from mttl.models.utils import transfer_batch_to_device
from mttl.utils import remote_login

torch.set_float32_matmul_precision("high")

import torch


@dataclass
class ICVKMArguments(ExpertConfig):
    loss_function: str = "dcd"
    # set the following if you want to enable the NQA callback during training
    nqa_dataset: str = "sordonia/narrativeqa_sanitized"
    aggregation: str = "last"  # or "mean"


def train_km(training_args: ICVKMArguments):
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

    info_container = {}
    for it in range(10):
        # hooks = setup_hooks(model, info_container)

        # without grad, do a pass over the training dataset
        with torch.no_grad():
            datamodule = get_datamodule(training_args)
            pbar = tqdm(datamodule.train_dataloader())
            deltas = []
            for batch_idx, batch in enumerate(pbar):
                if batch_idx == 16:
                    break

                inputs = transfer_batch_to_device(batch, device)
                # document + small task prompt + task output (e.g. summary, or question and answer)
                input_ids = inputs["input_ids"]
                labels = inputs["labels"]
                attention_mask = inputs["attention_mask"]

                assert (
                    input_ids.size(0) == 1
                ), "TODO: change aggregation to handle multiple inputs"

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
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = torch.stack(
                    [
                        hidden_state[labels != -100, ...]
                        for hidden_state in outputs.hidden_states
                    ]
                )

                # Context forward pass
                info_container["use_context"] = False
                nc_outputs = model(
                    input_ids=nc_input_ids,
                    attention_mask=nc_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                nc_hidden_states = torch.stack(
                    [
                        hidden_state[nc_labels != -100, ...]
                        for hidden_state in nc_outputs.hidden_states
                    ]
                )

                # How to aggregate the sequence dimension ?
                # hidden_states are (num_layers, seq_len, hidden_size)
                if training_args.aggregation == "last":
                    hidden_states = hidden_states[:, -1]
                    nc_hidden_states = nc_hidden_states[:, -1]
                elif training_args.aggregation == "mean":
                    hidden_states = hidden_states.mean(dim=1)
                    nc_hidden_states = nc_hidden_states.mean(dim=1)
                else:
                    raise ValueError(
                        f"Aggregation {training_args.aggregation} not supported"
                    )

                # in ICV, delta = pos - neg, or target - pred
                deltas += [hidden_states - nc_hidden_states]
                # Compute DCD loss between inputs and outputs
                # and check whether iterating over time helps

            # Now, we have all the deltas, we can compute PCA
            breakpoint()
            n_layers, dim = deltas[0].shape
            deltas = torch.stack(deltas).flatten(-2)
            pca = PCA(n_components=1).to(deltas.device).fit(deltas.float())
            direction = (pca.components_.sum(dim=0, keepdim=True) + pca.mean_).mean(0)
            direction = direction.reshape(n_layers, dim)

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
                    output = torch.cat(module.output, dim=0)
                    A, B = closed_form_low_rank_solution(
                        input.float()[::20],
                        output.float()[::20],
                        module.config.lora_rank,
                        lambda_reg=100,
                    )

                    module.lora_a.data.copy_(A.data)
                    module.lora_b.data.copy_(B.data)

            remove_hooks(hooks)
            val_loss, rougeL = do_evaluation(
                datamodule, model, loss_function, evaluator
            )
            xx = 1


if __name__ == "__main__":
    args = ICVKMArguments.parse(raise_error=False)
    train_km(args)
