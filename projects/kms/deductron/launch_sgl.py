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
from accelerate.state import AcceleratorState
from accelerate.state import PartialState


state = PartialState()


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

    def __init__(self, model_name, seed, dp_size=2):
        self.model_name = model_name
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dp_size = dp_size
        self.start()
        SGLGenerator._instance = self

    @state.on_main_process
    def start(self):
        from sglang.utils import execute_shell_command, wait_for_server
        import os

        server_process = execute_shell_command(
            f"""python3 -m sglang.launch_server \
--model-path {self.model_name} \
--host 0.0.0.0 --port 30000 \
--log-level info \
--random-seed {self.seed} \
--base-gpu-id 2 \
--dp-size {self.dp_size}
"""
        )

        wait_for_server("http://localhost:30000")
        self.process = server_process


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


SGLGenerator(model_name="meta-llama/Llama-3.1-8B-Instruct", seed=42)
while True:
    time.sleep(5)
