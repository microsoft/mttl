import argparse
from contextlib import contextmanager
import gc
import math
from multiprocessing import Process
import os
import random
import signal
import subprocess
import sys
import time

import numpy as np
from projects.kms.deductron.algos.grpo import GRPO
import torch
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from accelerate import Accelerator
from projects.kms.deductron.algos.rft import RFT
from projects.kms.deductron.algos.rloo import RLOO
from projects.kms.deductron.data_utils import (
    MultiTensorDataset,
    chunk_text,
    get_dataloader,
    prepare_dataset,
)
from launch_sgl import SGLGenerator, is_server_up
from transformers.trainer_pt_utils import get_parameter_names
from projects.kms.deductron.ddp_utils import init_ddp, gather_and_concatenate
from projects.kms.deductron.gen_utils import GenerationBackend
from projects.kms.deductron.utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMP,
    AccumulatorDict,
    CosineWarmupScheduler,
    print_metrics,
    save_args,
    setup_output_directory,
)
import bitsandbytes as bnb
from projects.kms.deductron.ddp_utils import ddp_state
from sglang.utils import wait_for_server
import torch.multiprocessing as mp


models = {
    "q1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "ll8b": "meta-llama/Llama-3.1-8B-Instruct",
}


algos = {
    "rft": RFT,
    "rloo": RLOO,
    "grpo": GRPO,
}


def get_algo_kwargs(args, klass):
    parser = argparse.ArgumentParser()
    parser = klass.add_parser_args(parser)
    argnames = [action.dest for action in parser._actions]
    return {k: v for k, v in vars(args).items() if k in argnames}


def train(local_rank, args):
    if local_rank == 0:
        print("=============================")
        print("Bringing up SGL server")
        generator = SGLGenerator(models[args.m], args.s)

    # wait for generator to be up!
    GenerationBackend.init(args.b, model_name=models[args.m], seed=args.s)
    init_ddp(local_rank, args.P)

    GenerationBackend.get()

    # output directory
    torch.manual_seed(args.s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.s)
    np.random.seed(args.s)
    random.seed(args.s)

    if args.o == "auto":
        # create a timestamp
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.o = f"runs_output/{args.a}_{args.m}_{args.s}_{timestamp}"

    max_epochs = args.epc
    max_off_epochs = args.offepc
    # total batch size across devices
    onl_batch_size = args.onlbsz * ddp_state.num_processes
    # this is *per device*
    off_batch_size = args.offbsz
    inn_batch_size = args.innbsz
    acc_steps = off_batch_size // inn_batch_size

    assert (
        onl_batch_size % ddp_state.num_processes == 0
    ), "Batch size must be divisible by the number of GPUs!"

    if ddp_state.is_main_process:
        setup_output_directory(args.o)
        save_args(args, args.o)

    algo = algos[args.a](
        models[args.m],
        k=args.k,
        temperature=args.t,
        max_tokens=args.maxtok,
        device=ddp_state.device,
        task=args.task,
        **get_algo_kwargs(args, algos[args.a]),
    )
    algo.model.gradient_checkpointing_enable()
    algo.model = DDP(
        algo.model,
        device_ids=[ddp_state.local_process_index],
        output_device=ddp_state.local_process_index,
    )

    decay_parameters = get_parameter_names(algo.model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in algo.model.named_parameters() if n in decay_parameters
            ],
            "weight_decay": 1e-6,
        },
        {
            "params": [
                p for n, p in algo.model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        lr=args.lr,
    )
    ddp_state.print(torch.cuda.mem_get_info())

    if args.a == "rloo":
        num_ex = onl_batch_size * args.k
    else:
        num_ex = onl_batch_size

    total_steps = (max_epochs * max_off_epochs * num_ex) // (
        acc_steps * ddp_state.num_processes
    )
    scheduler = CosineWarmupScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=1e-3 * args.lr,
        warmup_steps=0.1 * total_steps,
        max_steps=total_steps,
    )
    ddp_state.print("Scheduler set! Total steps:", total_steps)

    with ddp_state.main_process_first():
        train_dataset, val_dataset, test_dataset = prepare_dataset(
            args.dataset, algo.tokenizer, block_size=2048
        )

    sample_indices = np.random.choice(len(val_dataset), 32, replace=False)
    val_queries = [val_dataset[int(i)]["source"] for i in sample_indices]
    val_labels = [val_dataset[int(i)]["label"] for i in sample_indices]

    training_stats = []
    # do validation
    with ddp_state.split_between_processes(
        list(zip(val_queries, val_labels))
    ) as partial_batch:
        algo.model.eval()
        part_queries, part_labels = zip(*partial_batch)
        val_requests = algo.compute_rewards(part_queries, part_labels, k=1)
        val_reward = torch.tensor(
            np.mean([r.reward for r in val_requests]), device=ddp_state.device
        )

    torch.distributed.all_reduce(val_reward, op=torch.distributed.ReduceOp.AVG)
    val_reward = val_reward.item()
    training_stats.append(
        {
            "val_reward": val_reward,
        }
    )
    ddp_state.wait_for_everyone()
    ddp_state.print("Initial reward:", val_reward)

    assert (
        onl_batch_size >= args.subs
    ), "Batch size cannot be smaller than the dataset size!"

    global_step = 0
    for epoch in range(max_epochs):
        epoch_stats = AccumulatorDict()

        sample_indices = np.random.choice(
            len(train_dataset), onl_batch_size, replace=False
        )
        queries_batch = [train_dataset[int(i)]["source"] for i in sample_indices]
        labels_batch = [train_dataset[int(i)]["label"] for i in sample_indices]

        # create dataset out of gathered episodes
        with ddp_state.split_between_processes(
            list(zip(queries_batch, labels_batch))
        ) as partial_batch:
            part_queries, part_labels = zip(*partial_batch)
            episode_data = algo.gather_episodes(part_queries, part_labels)

        epoch_dataset = MultiTensorDataset(*episode_data)
        dataloader = get_dataloader(epoch_dataset, inn_batch_size)

        ddp_state.print("====================================")
        ddp_state.print("Beginning updating the policy")
        ddp_state.print("Length of the dataset:", len(epoch_dataset))

        torch.save(
            episode_data,
            f"{args.o}/episode_data_{ddp_state.local_process_index}_{epoch}.pt",
        )

        # offline steps
        train_iterator = iter(dataloader)
        off_sampler = dataloader.sampler
        off_steps = max_off_epochs * len(dataloader)
        off_epoch = 0

        for off_epoch in range(max_off_epochs):
            algo.model.train()
            optimizer.zero_grad()
            train_iterator = iter(dataloader)

            for micro_step, batch in enumerate(
                tqdm(
                    train_iterator,
                    total=len(dataloader),
                    desc="Offline epoch {}".format(off_epoch),
                    disable=not ddp_state.is_main_process,
                )
            ):
                loss_batch = 0
                batch = [b.to(ddp_state.device) for b in batch]

                algo.model.require_backward_grad_sync = micro_step + 1 == acc_steps
                loss = algo.compute_loss(batch)

                (loss / acc_steps).backward()
                epoch_stats.accumulate("loss", loss.item())
                del batch
                del loss
                torch.cuda.empty_cache()

                if micro_step > 0 and (micro_step + 1) % acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(algo.model.parameters(), 1.0)
                    optimizer.step()
                    torch.cuda.synchronize()
                    scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

                global_step += 1

            ddp_state.print(
                f"Epoch: {epoch}, Off Epoch: {off_epoch}, "
                f"Step: {global_step}, {algo.__class__.__name__}, "
                f"Loss: {epoch_stats.mean('loss'):.4f}, "
                f"Lr: {scheduler.get_last_lr()[0]:.6f}"
            )

        del dataloader
        del episode_data
        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        ddp_state.wait_for_everyone()

        # do validation
        with ddp_state.split_between_processes(
            list(zip(val_queries, val_labels))
        ) as partial_batch:
            algo.model.eval()
            part_queries, part_labels = zip(*partial_batch)
            val_requests = algo.compute_rewards(part_queries, part_labels, k=1)
            val_reward = torch.tensor(
                np.mean([r.reward for r in val_requests]), device=ddp_state.device
            )

        torch.distributed.all_reduce(val_reward, op=torch.distributed.ReduceOp.AVG)
        val_reward = val_reward.item()
        ddp_state.wait_for_everyone()

        if ddp_state.is_main_process:
            # append a bunch of training stats
            training_stats.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    "val_reward": val_reward,
                    **epoch_stats.get(),
                    **algo.stats.get(),
                }
            )

            # save stats
            with open(f"{args.o}/training_stats.json", "w") as f:
                import json

                json.dump(training_stats, f)

            # save best model
            if epoch == 0 or training_stats[-1]["val_reward"] >= np.max(
                [t["val_reward"] for t in training_stats[:-1]]
            ):
                if ddp_state.num_processes > 1:
                    algo.model.module.save_pretrained(f"{args.o}/model")
                else:
                    algo.model.save_pretrained(f"{args.o}/model")
                algo.tokenizer.save_pretrained(f"{args.o}/model")

            print("====================================")
            print(f"Epoch {epoch}, Avg Reward: {training_stats[-1]['val_reward']:.4f}")
            print(training_stats[-1])
            print(
                "Reward So Far:",
                print_metrics([t.get("avg_reward", 0) for t in training_stats]),
            )
            print(
                "Val So Far:",
                print_metrics([t.get("val_reward", 0) for t in training_stats]),
            )
            print("====================================")

            # update the weights of the data generator after the epoch
            GenerationBackend.get().load_weights(algo.model)

        ddp_state.wait_for_everyone()

    if ddp_state.local_process_index == 0:
        generator.shutdown()
    ddp_state.wait_for_everyone()


@contextmanager
def patch_environment(**kwargs):
    existing_vars = {}
    for key, value in kwargs.items():
        key = key.upper()
        if key in os.environ:
            existing_vars[key] = os.environ[key]
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key in kwargs:
            key = key.upper()
            if key in existing_vars:
                # restore previous value
                os.environ[key] = existing_vars[key]
            else:
                os.environ.pop(key, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Model name")
    parser.add_argument("-o", type=str, help="Output directory")
    parser.add_argument("-s", type=int, help="Seed", default=42)
    parser.add_argument("-a", type=str, help="Algorithm")
    parser.add_argument("--dataset", type=str, help="Dataset", default="nqa")
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-6)
    parser.add_argument("--epc", type=int, help="Number of epochs", default=20)
    parser.add_argument("--onlbsz", type=int, help="Online batch size", default=32)
    parser.add_argument(
        "--offepc", type=int, help="Number of offline epochs", default=4
    )
    parser.add_argument(
        "--offbsz", type=int, help="Offline batch size **per device**", default=8
    )
    parser.add_argument(
        "--innbsz", type=int, help="Inner/Grad accumulation batch size", default=1
    )
    parser.add_argument("-k", type=int, help="Number of samples", default=5)
    parser.add_argument("-t", type=float, help="Temperature", default=DEFAULT_TEMP)
    parser.add_argument(
        "--maxtok", type=int, help="Number of tokens", default=DEFAULT_MAX_TOKENS
    )
    parser.add_argument("-b", type=str, help="Backend", default="sglc")
    parser.add_argument("--tpsz", type=int, help="Tensor parallel size", default=1)
    parser.add_argument("--ss", type=str, help="Math subset to consider", default="all")
    parser.add_argument("--subs", type=int, help="Subsample examples", default=-1)
    parser.add_argument(
        "--task",
        type=str,
        help="Type of task to solve",
        default="s_ae",
        choices=["s_ae", "s_ncp"],
    )
    parser.add_argument("-d", type=str, help="Run description", default=None)
    parser.add_argument("-P", type=int, help="Num Processes", default=1)
    parser.add_argument("--dmpepc", action="store_true", help="Dumps every epoch")

    # parse known args first
    partial_args, unknown_args = parser.parse_known_args()

    # conditionally add algorithm-specific args
    algos[partial_args.a].add_parser_args(parser)

    # Parse final args
    final_args = parser.parse_args()

    if is_server_up():
        raise ValueError("Terminate previous server before starting!")

    os.environ["OMP_NUM_THREADS"] = "1"
    mp.spawn(train, args=(final_args,), nprocs=final_args.P, join=True)
