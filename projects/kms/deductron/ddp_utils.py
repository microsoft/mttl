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
    process_group: Any = None
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_world_size: int = 1
    is_master: bool = True
    device: str = "cuda"


ddp_state = DDPState()


def init_ddp():
    global ddp_state

    ddp_state.ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp_state.ddp:
        print("Running in DDP mode!")
        ddp_state.process_group = init_process_group(group_name="main_group")
        ddp_state.ddp_rank = int(os.environ["RANK"])
        ddp_state.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_state.ddp_world_size = int(os.environ["WORLD_SIZE"])
        ddp_state.device = f"cuda:{ddp_state.ddp_local_rank}"
        torch.cuda.set_device(ddp_state.device)
        ddp_state.is_master = ddp_state.ddp_rank == 0
    else:
        print("Running in non-DDP mode!")


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
        if ddp_state.is_master:
            # Execute the function on the master process
            result = func(*args, **kwargs)
            # Prepare empty list to receive the broadcasted object
            obj_list = [result]
            # Receive the broadcasted object list from master
            broadcast_object_list(obj_list, src=0, group=ddp_state.process_group)
            # Retrieve the result
            result = obj_list[0]
            return result
        else:
            # Prepare empty list to receive the broadcasted object
            obj_list = [None]
            # Receive the broadcasted object list from master
            broadcast_object_list(obj_list, src=0, group=ddp_state.process_group)
            # Retrieve the result
            result = obj_list[0]
            # Optionally, perform alternative actions or simply pass
            return result  # Or some default value if needed

    return wrapper
