import concurrent
import logging
import os
import sys
import threading
import time
from asyncio import threads
from functools import partial
from typing import List, Optional

import docker
import psutil
import requests
import torch
import tqdm
from docker.errors import APIError, NotFound
from transformers import AutoTokenizer
from vllm.worker.worker import Worker

from projects.kms.deductron.ddp_utils import ddp_state
from projects.kms.deductron.utils import DEFAULT_MAX_TOKENS, DEFAULT_TEMP


def wait_for_server_shutdown(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(1)

            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not die within timeout period")
        except requests.exceptions.RequestException:
            break


class SGLGenerator:
    _instance = None

    @classmethod
    def get(cls):
        assert cls._instance is not None, "SGLGenerator not initialized"
        return cls._instance

    def __init__(
        self,
        model_name,
        seed,
    ):
        self.model_name = model_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.available_devices = torch.cuda.device_count()
        self.world_size = ddp_state.ddp_world_size

        free_gpus = self.available_devices - self.world_size
        assert free_gpus > 0, "Not enough free GPUs"

        # even number of free gpus
        self.tp_size = (
            free_gpus if free_gpus == 1 or free_gpus % 2 == 0 else free_gpus - 1
        )
        self.devices_str = ",".join(
            list(
                map(
                    str,
                    range(self.world_size, self.world_size + self.tp_size),
                )
            ),
        )

        self.start()
        SGLGenerator._instance = self

    def shutdown(self):
        kill_sglang_container()

        print("Process killed, waiting for shutdown.")
        wait_for_server_shutdown("http://localhost:30000")

    def start(self):
        from sglang.utils import execute_shell_command, wait_for_server

        server_process = execute_shell_command(
            f"""docker run --gpus "device={self.devices_str}" \
--shm-size 32g \
-p 30000:30000 \
-v /tmp/saved_model:/root/saved_model \
--ipc=host \
lmsysorg/sglang:latest \
python3 -m sglang.launch_server \
    --model-path {self.model_name} \
    --host 0.0.0.0 --port 30000 \
    --log-level warning \
    --random-seed {self.seed} \
    --tp {self.tp_size}
"""
        )

        wait_for_server("http://localhost:30000")
        self.process = server_process

    def load_weights(self, model):
        import requests

        if hasattr(model, "module"):
            model = model.module

        model.save_pretrained("/tmp/saved_model")
        response = requests.post(
            "http://localhost:30000/update_weights_from_disk",
            json={"model_path": "/root/saved_model"},
        )
        assert response.json()["success"] is True

    def chat(
        self,
        messages,
        temperature=DEFAULT_TEMP,
        top_p=1.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        n=1,
        return_finished=False,
    ):
        import requests

        prompts = [
            self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=message[-1]["role"] == "user",
                continue_final_message=message[-1]["role"] == "assistant",
                tokenize=False,
            )
            for message in messages
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(
                        partial(send_request, temperature, top_p, max_tokens, n),
                        prompts,
                    ),
                    total=len(prompts),
                )
            )

        outputs = []
        finished = []
        for outputs_, finished_ in results:
            outputs.append(outputs_)
            finished.append(finished_)

        if return_finished:
            return outputs, finished
        return outputs


def send_request(temperature, top_p, max_tokens, n, prompt):
    import requests

    response = requests.post(
        "http://localhost:30000/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "n": n,
                "top_p": top_p,
            },
        },
    )
    response = response.json()
    if type(response) != list:
        response = [response]

    outputs = [r["text"] for r in response]
    finished = [r["meta_info"]["finish_reason"]["type"] == "stop" for r in response]
    return outputs, finished


def kill_sglang_container():
    try:
        client = docker.from_env()
    except Exception as e:
        print(f"Failed to initialize Docker client: {e}")
        sys.exit(1)

    containers = client.containers.list(all=True)
    sglang_containers = []
    for container in containers:
        try:
            if (
                container.attrs["Config"]["Cmd"]
                and "sglang.launch_server" in container.attrs["Config"]["Cmd"]
            ):
                sglang_containers.append(container)
        except:
            continue

    for container in sglang_containers:
        container = client.containers.get(container.id)
        if container:
            try:
                container.stop()
                print(f"Container '{container.id}' stopped gracefully.")
            except APIError as e:
                container.kill()
