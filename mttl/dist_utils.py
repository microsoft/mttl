import os
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistributedState:
    ddp: bool = False
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_world_size: int = 1
    master_process: bool = True


ddp_state = DistributedState()


def init_ddp():
    global ddp_state

    # assuming nccl here
    if (
        int(os.environ.get("RANK", -1)) != -1
        and int(os.environ.get("LOCAL_RANK", -1)) != -1
    ):
        dist.init_process_group(backend="nccl")

        ddp_state.ddp = True
        ddp_state.ddp_rank = int(os.environ.get("RANK", 0))
        ddp_state.ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ddp_state.ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp_state.master_process = ddp_state.ddp_rank == 0


init_ddp()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist_avail_and_initialized():
    return ddp_state.ddp


def get_data_sampler(dataset):
    if is_dist_avail_and_initialized():
        return torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        return None


def distributed_mean(metrics: List[float], device: torch.device) -> float:
    count = len(metrics)
    metric = np.sum(metrics)
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
        metric = torch.tensor(metric, dtype=torch.float32, device=device)
        count = torch.tensor(count, dtype=torch.long, device=device)
        dist.all_reduce(metric, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        value = metric.item() / count.item()
    else:
        value = float(metric) / count
    return value


def get_device():
    if is_dist_avail_and_initialized():
        return "cuda:{:d}".format(get_local_rank())
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_local_rank():
    return ddp_state.ddp_local_rank


def get_world_size():
    return ddp_state.ddp_world_size


def get_rank():
    return ddp_state.ddp_rank


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
