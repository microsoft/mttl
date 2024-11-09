import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist

# assuming nccl here
if int(os.environ.get("RANK", -1)) != -1:
    dist.init_process_group(backend="nccl")

    ddp = True
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    master_process = ddp_rank == 0
else:
    ddp = False
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True


def is_dist_avail_and_initialized():
    return ddp


def get_data_sampler(dataset):
    if is_dist_avail_and_initialized():
        return torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        return None


def distributed_mean(metrics: List[float], device: torch.device) -> float:
    count = len(metrics)
    metric = np.sum(metrics)
    if is_dist_avail_and_initialized():
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
    return ddp_local_rank


def get_world_size():
    return ddp_world_size


def get_rank():
    return ddp_rank


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
