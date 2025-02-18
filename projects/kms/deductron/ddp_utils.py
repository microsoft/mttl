from contextlib import contextmanager
import functools
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed
from torch.distributed import broadcast_object_list, init_process_group

# total number of gpus
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPUs found!")

# deactivate this to avoid painful messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class DDPState:
    ddp: bool = False
    num_processes: int = 1
    process_group: Any = None
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_world_size: int = 1
    device: str = "cuda:0"
    is_main_process: bool = True
    local_process_index: int = 0
    process_index: int = 0

    def wait_for_everyone(self):
        torch.distributed.barrier()

    def on_main_process(self, function):
        if self.is_main_process:
            return function
        return lambda *args, **kwargs: None

    @contextmanager
    def main_process_first(self):
        if not self.is_main_process:
            self.wait_for_everyone()

        yield

        if self.is_main_process:
            self.wait_for_everyone()

    def print(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    @contextmanager
    def split_between_processes(self, inputs):
        if self.num_processes == 1:
            yield inputs
            return

        length = len(inputs)
        assert len(inputs) % self.num_processes == 0

        num_samples_per_process = length // self.num_processes
        start_index = self.process_index * num_samples_per_process
        end_index = start_index + num_samples_per_process

        yield inputs[start_index:end_index]


ddp_state = DDPState()


def init_ddp(local_rank=0, world_size=1):
    global ddp_state
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    ddp_state.process_group = init_process_group(
        "nccl",
        world_size=world_size,
        rank=local_rank,
    )
    ddp_state.ddp_rank = int(local_rank)
    ddp_state.ddp_local_rank = int(local_rank)
    ddp_state.ddp_world_size = int(world_size)
    ddp_state.local_process_index = int(local_rank)
    ddp_state.process_index = int(local_rank)
    ddp_state.num_processes = world_size
    ddp_state.device = f"cuda:{ddp_state.ddp_local_rank}"
    torch.cuda.set_device(ddp_state.device)
    ddp_state.is_main_process = ddp_state.ddp_rank == 0
    print("Running in DDP mode!")
    print(ddp_state)


def rank_zero_only(func):
    """
    Decorator that ensures the decorated function is only executed by the master process (rank 0).
    Non-master processes will wait until the master process completes the function.

    Args:
        func (callable): The function to decorate.

    Returns:
        callable: The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ddp_state.is_main_process:
            # Execute the function on the master process
            result = func(*args, **kwargs)
            # Prepare empty list to receive the broadcasted object
            obj_list = [result]
            # Receive the broadcasted object list from master
            torch.distributed.broadcast_object_list(obj_list)
            # Retrieve the result
            result = obj_list[0]
            return result
        else:
            # Prepare empty list to receive the broadcasted object
            obj_list = [None]
            # Receive the broadcasted object list from master
            torch.distributed.broadcast_object_list(obj_list)
            # Retrieve the result
            result = obj_list[0]
            # Optionally, perform alternative actions or simply pass
            return result  # Or some default value if needed

    return wrapper


def gather_and_concatenate(data, dim=0):
    world_size = ddp_state.num_processes

    gathered_data = []
    for tensor in data:
        # Prepare a list to hold one tensor per process
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(
            gathered_tensors, tensor
        )  # Each rank populates gathered_tensors
        gathered_data.append(torch.cat(gathered_tensors, dim=dim))
    return gathered_data
