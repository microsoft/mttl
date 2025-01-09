from datetime import timedelta
from typing import List, Optional

import ray
import torch
import torch.distributed
import vllm
from openai import OpenAI
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_world_group,
    init_model_parallel_group,
)
from vllm.executor.gpu_executor import GPUExecutor
from vllm.worker.worker import Worker

from projects.kms.deductron.utils import DEFAULT_MAX_TOKENS, DEFAULT_TEMP
from accelerate.state import PartialState


state = PartialState()


class VLLMGenerator:
    _instance = None

    @classmethod
    def get(cls):
        assert cls._instance is not None, "VLLMGenerator not initialized"
        return cls._instance

    def __init__(
        self,
        model_name,
        seed,
        max_num_seqs=32,
        gpu_memory_utilization=0.8,
    ):
        from accelerate.state import AcceleratorState

        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model_name = model_name
        self.seed = seed

        vllm_single_gpu_patch()
        self.start()
        VLLMGenerator._instance = self

    @state.on_main_process
    def start(self):
        self.process = LLM(
            self.model_name,
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            device=f"cuda:{state.num_processes}",
            seed=self.seed,
        )

    @state.on_main_process
    def load_weights(self, model):
        if isinstance(model, DDP):
            model = model.module
        else:
            model = model
        self.process.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
            model.named_parameters()
        )

    @state.on_main_process
    def shutdown(self):
        del self.process

    def chat(
        self,
        messages,
        temperature=DEFAULT_TEMP,
        top_p=1.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        n=1,
        return_finished=False,
    ):
        """We need to handle differently messages that end with a 'user' role and 'assistant' role:

        * We consider that the last message needs to be continued if it ends with an 'assistant' role.
        * We consider that the last message needs a generation prompt if it ends with a 'user' role.

        Return responses.
        """

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            top_p=top_p,
        )

        # handle automatically continuing the final message or not
        continue_messages = [
            i for i, m in enumerate(messages) if m[-1]["role"] == "assistant"
        ]
        standard_messages = [
            i for i in range(len(messages)) if i not in continue_messages
        ]
        if continue_messages:
            continue_outputs = self.process.chat(
                [messages[i] for i in continue_messages],
                sampling_params=params,
                continue_final_message=True,
                add_generation_prompt=False,
            )
        else:
            continue_outputs = []

        if standard_messages:
            outputs = self.process.chat(
                [messages[i] for i in standard_messages],
                sampling_params=params,
            )
        else:
            outputs = []

        for i, o in zip(continue_messages, continue_outputs):
            outputs.insert(i, o)

        responses = [[k.text for k in o.outputs] for o in outputs]
        if return_finished:
            finished = [[k.finish_reason == "stop" for k in o.outputs] for o in outputs]
            return responses, finished
        return responses


def custom_initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.
    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = 1
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    # global _TP
    assert (
        vllm.distributed.parallel_state._TP is None
    ), "tensor model parallel group is already initialized"
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        )
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    vllm.distributed.parallel_state._TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
    )

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    # global _PP
    assert (
        vllm.distributed.parallel_state._PP is None
    ), "pipeline model parallel group is already initialized"
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    vllm.distributed.parallel_state._PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, use_custom_allreduce=False
    )


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[[0]],  # SingleGPULLM logic: only use a single GPU
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=False,
        use_tpu_communicator=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
    )


def _init_executor(self) -> None:
    """Initialize the worker and load the model."""
    assert self.parallel_config.world_size == 1, "GPUExecutor only supports single GPU."

    self.driver_worker = self._create_worker(local_rank=self.device_config.device.index)
    self.driver_worker.init_device()
    self.driver_worker.load_model()


# monkey patch the function
def vllm_single_gpu_patch():
    vllm.distributed.parallel_state.init_world_group = init_world_group
    vllm.distributed.parallel_state.initialize_model_parallel = (
        custom_initialize_model_parallel
    )
    GPUExecutor._init_executor = _init_executor
