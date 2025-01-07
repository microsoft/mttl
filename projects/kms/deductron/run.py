import argparse
import gc
import random

import deepspeed
import numpy as np
import torch
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from projects.kms.deductron.algos.rft import RFT
from projects.kms.deductron.algos.rloo import RLOO
from projects.kms.deductron.data_utils import (
    MultiTensorDataset,
    chunk_text,
    get_dataloader,
    prepare_nqa_dataset,
)
from projects.kms.deductron.ddp_utils import ddp_state, init_ddp
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

models = {
    "q1.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "ll8b": "meta-llama/Llama-3.1-8B-Instruct",
}

algos = {
    "rft": RFT,
    "rloo": RLOO,
}


def get_algo_kwargs(args, klass):
    parser = argparse.ArgumentParser()
    parser = klass.add_parser_args(parser)
    argnames = [action.dest for action in parser._actions]
    return {k: v for k, v in vars(args).items() if k in argnames}


def train(args):
    init_ddp()

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

    setup_output_directory(args.o)
    save_args(args, args.o)

    algo = algos[args.a](
        models[args.m],
        k=args.k,
        temperature=args.t,
        max_tokens=args.maxtok,
        device=ddp_state.device,
        task_generator="summary",
        reward_function="logprobs",
        **get_algo_kwargs(args, algos[args.a]),
    )

    max_epochs = args.epc
    max_off_epochs = args.offepc
    onl_batch_size = args.bsz
    off_batch_size = args.offbsz
    inn_batch_size = args.innbsz
    total_steps = max_epochs * max_off_epochs

    assert (
        onl_batch_size % ddp_state.ddp_world_size == 0
    ), "Batch size must be divisible by the number of GPUs!"

    training_stats = []
    optimizer = torch.optim.AdamW(algo.model.parameters(), lr=args.lr)
    scheduler = CosineWarmupScheduler(
        optimizer,
        max_lr=args.lr,
        min_lr=1e-3 * args.lr,
        warmup_steps=0.1 * total_steps,
        max_steps=total_steps,
    )
    if ddp_state.ddp:
        algo.model = DDP(algo.model, device_ids=[ddp_state.ddp_local_rank])

    # Initialize the VLLM model after the DDP initialization
    if ddp_state.is_master:
        GenerationBackend.init(args.b, model_name=models[args.m], seed=args.s)

    dataset = prepare_nqa_dataset(algo.tokenizer, block_size=1024)

    # Form a small subset of the training data for evaluation
    train_subsample = np.random.choice(len(dataset), 100, replace=False)
    train_queries = [dataset[int(i)]["source"] for i in train_subsample]
    train_labels = [dataset[int(i)]["label"] for i in train_subsample]

    assert (
        onl_batch_size >= args.subs
    ), "Batch size cannot be smaller than the dataset size!"

    global_step = 0
    for epoch in range(max_epochs):
        epoch_stats = AccumulatorDict()

        sample_indices = np.random.choice(len(dataset), onl_batch_size, replace=False)
        queries_batch = [dataset[int(i)]["source"] for i in sample_indices]
        labels_batch = [dataset[int(i)]["label"] for i in sample_indices]

        # create dataset out of gathered episodes
        episode_data = algo.gather_episodes(queries_batch, labels_batch)
        epoch_dataset = MultiTensorDataset(*episode_data)
        dataloader = get_dataloader(
            epoch_dataset, off_batch_size // ddp_state.ddp_world_size
        )

        if ddp_state.is_master:
            print("====================================")
            print("Beginning updating the policy")
            print("Length of the dataset:", len(epoch_dataset))

        # offline steps
        train_iterator = iter(dataloader)
        off_sampler = dataloader.sampler
        off_steps = max_off_epochs * len(dataloader)
        off_epoch = 0

        for off_epoch in range(max_off_epochs):
            train_iterator = iter(dataloader)

            for batch in tqdm(
                train_iterator,
                total=len(dataloader),
                desc="Offline epoch {}".format(off_epoch),
            ):
                loss_batch = 0
                for step in range(0, batch[0].shape[0], inn_batch_size):
                    loss = algo.compute_loss(
                        [x[step : step + inn_batch_size].to(ddp_state.device) for x in batch]
                    )
                    loss = loss / off_batch_size
                    loss.backward()
                    loss_batch += loss.item()
                    torch.cuda.empty_cache()
                    del loss

                epoch_stats.accumulate("loss", loss_batch)
                torch.nn.utils.clip_grad_norm_(algo.model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                global_step += 1

            if ddp_state.is_master:
                print(
                    f"Epoch: {epoch}, Off Epoch: {off_epoch}, "
                    f"Step: {global_step}, {algo.__class__.__name__}, "
                    f"Loss: {epoch_stats.mean('loss'):.4f}, "
                    f"Lr: {scheduler.get_last_lr()[0]:.6f}"
                )

            scheduler.step()

        del dataloader

        gc.collect()
        torch.cuda.empty_cache()

        # update the weights of the data generator after the epoch
        if ddp_state.is_master:
            # update our data generator
            GenerationBackend.get().load_weights(algo.model)

        # append a bunch of training stats
        training_stats.append(
            {
                "epoch": epoch,
                "step": global_step,
                "lr": scheduler.get_last_lr()[0],
                **epoch_stats.get(),
                **algo.stats.get(),
            }
        )

        if ddp_state.is_master:
            print("====================================")
            print(f"Epoch {epoch}, Avg Reward: {training_stats[-1]['avg_reward']:.4f}")
            print(training_stats[-1])
            print(
                "Reward So Far:",
                print_metrics([t.get("avg_reward", 0) for t in training_stats]),
            )
            print("====================================")

            # save data
            torch.save(episode_data, f"{args.o}/episode_data_{epoch}.pt")

            # save stats
            with open(f"{args.o}/training_stats.json", "w") as f:
                import json

                json.dump(training_stats, f)

            # save best model
            if epoch == 0 or training_stats[-1]["avg_reward"] >= np.max(
                [t["avg_reward"] for t in training_stats[:-1]]
            ):
                if ddp_state.ddp:
                    algo.model.module.save_pretrained(f"{args.o}/model")
                else:
                    algo.model.save_pretrained(f"{args.o}/model")
                algo.tokenizer.save_pretrained(f"{args.o}/model")

        if ddp_state.ddp:
            torch.distributed.barrier()

    if ddp_state.is_master:
        GenerationBackend.get().shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, help="Model name")
    parser.add_argument("-o", type=str, help="Output directory")
    parser.add_argument("-s", type=int, help="Seed", default=42)
    parser.add_argument("-a", type=str, help="Algorithm")
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-5)
    parser.add_argument("--epc", type=int, help="Number of epochs", default=20)
    parser.add_argument("--bsz", type=int, help="Online batch size", default=32)
    parser.add_argument(
        "--offepc", type=int, help="Number of offline epochs", default=4
    )
    parser.add_argument("--offbsz", type=int, help="Offline batch size", default=8)
    parser.add_argument(
        "--innbsz", type=int, help="Inner/Grad accumulation batch size", default=1
    )
    parser.add_argument("-k", type=int, help="Number of samples", default=5)
    parser.add_argument("-t", type=float, help="Temperature", default=DEFAULT_TEMP)
    parser.add_argument(
        "--maxtok", type=int, help="Number of tokens", default=DEFAULT_MAX_TOKENS
    )
    parser.add_argument("-b", type=str, help="Backend", default="sgl")
    parser.add_argument("--tpsz", type=int, help="Tensor parallel size", default=1)
    parser.add_argument("--ss", type=str, help="Math subset to consider", default="all")
    parser.add_argument("--subs", type=int, help="Subsample examples", default=-1)
    parser.add_argument(
        "--fast", action="store_true", help="Use fast mode (no eval on epoch 0)"
    )
    parser.add_argument("-d", type=str, help="Run description", default=None)
    parser = deepspeed.add_config_arguments(parser)

    # parse known args first
    partial_args, unknown_args = parser.parse_known_args()

    # conditionally add algorithm-specific args
    algos[partial_args.a].add_parser_args(parser)

    # Parse final args
    final_args = parser.parse_args()
    train(final_args)
