import asyncio
import json
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from time import sleep

from azure.storage.blob import BlobClient
from azure.storage.blob.aio import BlobClient as AsyncBlobClient
from azure.storage.blob.aio import BlobLeaseClient
from filelock import AsyncFileLock

from mttl.models.library.backend_engine import BlobStorageEngine

got_sigterm = False
got_exit_signal = False


def handle_exit_signal(signum, frame):
    print(f"Received exit signal {signum}, initiating graceful shutdown...")
    global got_exit_signal
    got_exit_signal = True


class JobQueue:
    def __init__(self, job_file, lock_file, n_retries, use_blob_storage=False):
        self.job_file = job_file
        # self.lock_file = lock_file
        self.queue_file = lock_file.replace(".lock", "_queue.json")

        self.n_retries = n_retries
        self.use_blob_storage = use_blob_storage
        if use_blob_storage:
            self.blob_engine = BlobStorageEngine()
        print(f"queue file : {self.queue_file}")

    def _format_blob_path(self, path):
        # Ensure the path is a valid URL for Azure Blob Storage
        formatted_path = path.replace(os.sep, "/").replace(" ", "%20")
        if len(formatted_path) > 1024:
            raise ValueError("The path is too long for Azure Blob Storage.")
        return formatted_path

    async def initialize_queue_file(self):
        if self.use_blob_storage:
            await self._initialize_queue_file_blob()
        else:
            await self._initialize_queue_file_local()

    async def _initialize_queue_file_local(self):
        # This function ensures the queue file is in the correct format.
        if not os.path.exists(self.queue_file):
            # copy the original job file to the queue file
            subprocess.run(["cp", self.job_file, self.queue_file])

            # convert the queue_file to the desired format
            async with AsyncFileLock(self.lock_file):
                with open(self.queue_file, "r") as f:
                    data = json.load(f)

                # If data is a dict with train/valid/test keys, flatten it.
                if isinstance(data, dict):
                    doc_ids = []
                    for value in data.values():
                        doc_ids += value
                else:
                    # If data is already a single list, just use it directly.
                    doc_ids = data

                # Convert to list of {"doc_id": ..., "status": "queued", "retries": 0}
                tasks = [
                    {"doc_id": doc_id, "status": "queued", "retries": 0}
                    for doc_id in doc_ids
                ]

                with open(self.queue_file, "w") as f:
                    json.dump(tasks, f, indent=2)

    async def _initialize_queue_file_blob(self):
        # This function ensures the queue file is in the correct format.

        if not self.blob_engine._blob_exists(self.queue_file):
            # copy the original job file to the queue file
            # convert the queue_file to the desired format
            print("gonna acquire lease")
            async with self._blob_lock(self.queue_file):
                print("opening file")
                data = json.load(open(self.job_file, "r"))

                # If data is a dict with train/valid/test keys, flatten it.
                if isinstance(data, dict):
                    doc_ids = []
                    for value in data.values():
                        doc_ids += value
                else:
                    # If data is already a single list, just use it directly.
                    doc_ids = data

                # Convert to list of {"doc_id": ..., "status": "queued", "retries": 0}
                tasks = [
                    {"doc_id": doc_id, "status": "queued", "retries": 0}
                    for doc_id in doc_ids
                ]

                print(f"Uploading {self.job_file} to {self.queue_file}")
                print("Uploaded job file : ", self.job_file)

                filename = self.queue_file.split("/")[-1]
                await self.blob_engine._async_upload_blob(
                    repo_id=self.queue_file,
                    filename=filename,
                    buffer=json.dumps(tasks, indent=2).encode("utf-8"),
                )

    async def get_next_job(self):
        if self.use_blob_storage:
            return await self._get_next_job_blob()
        else:
            return await self._get_next_job_local()

    async def _get_next_job_local(self):
        async with AsyncFileLock(self.lock_file):
            if not os.path.exists(self.queue_file):
                return None

            with open(self.queue_file, "r") as f:
                tasks = json.load(f)

            doc_id = None
            # Find the first job that is still queued
            for task in tasks:
                if task["status"] == "queued":
                    # Mark as running
                    task["status"] = "running"
                    doc_id = task["doc_id"]
                    break

            # Now, let's see if are some crashed of failed jobs we can retry
            for task in tasks:
                if task["status"] in ["crashed", "failed"]:
                    if task["retries"] < self.n_retries:
                        # Mark as running
                        task["status"] = "running"
                        task["retries"] += 1
                        doc_id = task["doc_id"]
                        break

            with open(self.queue_file, "w") as f:
                json.dump(tasks, f, indent=2)

            return doc_id

    async def _get_next_job_blob(self):
        async with self._blob_lock(self.queue_file):
            if not self.blob_engine._blob_exists(self.queue_file):
                raise FileNotFoundError(f"Queue file {self.queue_file} not found.")

            queue_path, queue_file = self.queue_file.split("/", 1)
            data = await self.blob_engine._async_download_blob(
                self.queue_file, self.queue_file.split("/")[-1]
            )
            print("DATA : ", data)
            tasks = json.load(open(data, "r"))

            doc_id = None
            # Find the first job that is still queued
            for task in tasks:
                if task["status"] == "queued":
                    # Mark as running
                    task["status"] = "running"
                    doc_id = task["doc_id"]
                    break

            # Now, let's see if there are some crashed or failed jobs we can retry
            for task in tasks:
                if task["status"] in ["crashed", "failed"]:
                    if task["retries"] < self.n_retries:
                        # Mark as running
                        task["status"] = "running"
                        task["retries"] += 1
                        doc_id = task["doc_id"]
                        break

            filename = self.queue_file.split("/")[-1]
            await self.blob_engine._async_upload_blob(
                repo_id=self.queue_file,
                filename=filename,
                buffer=json.dumps(tasks, indent=2).encode("utf-8"),
            )

            return doc_id

    async def update_job_status(self, doc_id, new_status):
        if self.use_blob_storage:
            await self._update_job_status_blob(doc_id, new_status)
        else:
            await self._update_job_status_local(doc_id, new_status)

    async def _update_job_status_local(self, doc_id, new_status):
        async with AsyncFileLock(self.lock_file):
            if not os.path.exists(self.queue_file):
                return

            with open(self.queue_file, "r") as f:
                tasks = json.load(f)

            # Update the status of the job with the given doc_id
            for task in tasks:
                if task["doc_id"] == doc_id:
                    # For other statuses, set the status directly
                    task["status"] = new_status
                    break

            with open(self.queue_file, "w") as f:
                json.dump(tasks, f, indent=2)

    async def _update_job_status_blob(self, doc_id, new_status):
        async with self._blob_lock(self.queue_file):
            if not self.blob_engine._blob_exists(self.queue_file):
                return

            queue_path, queue_file = self.queue_file.split("/", 1)
            data = await self.blob_engine._async_download_blob(
                self.queue_file, self.queue_file.split("/")[-1]
            )
            tasks = json.load(open(data, "r"))

            # Update the status of the job with the given doc_id
            for task in tasks:
                if task["doc_id"] == doc_id:
                    task["status"] = new_status
                    break

            filename = self.queue_file.split("/")[-1]
            await self.blob_engine._async_upload_blob(
                repo_id=self.queue_file,
                filename=filename,
                buffer=json.dumps(tasks, indent=2).encode("utf-8"),
            )

    @asynccontextmanager
    async def _blob_lock(self, blob_file, n_retries=10):
        """
        Acquire a lease on the referenced blob before operating.
        This mimics the azure sample's acquire_lease_on_blob_async approach.
        """
        lease_client = None
        n_retried = 0
        try:
            async with self.blob_engine._get_container_client(
                blob_file, use_async=True
            ) as container_client:
                while n_retried < n_retries:
                    try:
                        # Acquire a lease on the blob
                        lease_client = await container_client.acquire_lease(
                            lease_duration=15
                        )
                        yield
                        return
                    except Exception as e:
                        print("error ", e)
                        print("type ", type(e))
                        n_retried += 1
                        sleep(10)
        finally:
            if lease_client is not None:
                await lease_client.release()
            else:
                raise Exception(
                    f"Failed to acquire lease on {blob_file} after {n_retries} retries."
                )


async def run_job(gpu_id, doc_id, config_file, output_dir, train_file, config_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(env.get("GPU_OFFSET", 0)) + int(gpu_id))
    command = [
        "python",
        f"{train_file}.py",
        "-c",
        config_file,
        "-k",
        f"finetune_task_name={doc_id}",
        "-k",
        f"wandb_run_name={config_id}-{doc_id}",
        "-k",
        f"output_dir={output_dir}/{doc_id}",
    ]
    print(f"Assigning DOC_ID={doc_id} to GPU_ID={gpu_id}")

    async def stream_output(stream, prefix, job_name):
        job_name = str(job_name)[:10]
        job_name = job_name.ljust(10)
        try:
            async for line in stream:
                print(f"{prefix} <{job_name}>: {line.decode().strip()}")
        except asyncio.CancelledError:
            # Stream output cancelled, just return
            return

    process = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # If this task gets cancelled, we ensure to terminate the subprocess
    try:
        await asyncio.gather(
            stream_output(process.stdout, "STDOUT", doc_id),
            stream_output(process.stderr, "STDERR", doc_id),
        )
    except asyncio.CancelledError:
        # The job was cancelled, let's terminate the subprocess if it's still running
        if process.returncode is None:
            process.terminate()
            await process.wait()
        raise

    return_code = await process.wait()
    print(f"Process finished with exit code: {return_code} for DOC_ID={doc_id}")
    return doc_id, return_code


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Job Launcher")
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs")
    parser.add_argument("--jobs-per-gpu", type=int, required=True, help="Jobs per GPU")
    parser.add_argument(
        "--tasks-file", type=str, required=True, help="Path to the job file"
    )
    parser.add_argument("--config-id", type=str, required=True, help="Configuration ID")
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Training script filename without extension",
    )
    parser.add_argument(
        "--n-retries", type=int, default=0, help="Number of times to retry a failed job"
    )
    args = parser.parse_args()
    return (
        args.num_gpus,
        args.jobs_per_gpu,
        args.tasks_file,
        args.config_id,
        args.train_file,
        args.n_retries,
    )


def print_gpu_status(gpu_usage):
    print("=== GPU Status ===")
    for gpu_id, usage in gpu_usage.items():
        print(f"GPU {gpu_id}: {usage} job(s) running")
    print("==================")


async def main():
    num_gpus, jobs_per_gpu, job_file, config_id, train_file, n_retries = (
        parse_arguments()
    )
    config_file = f"configs/{config_id}.json"
    output_dir = f"/mnt/output/kms/{config_id}"

    # if running locally, change the output_dir
    if os.path.exists("/data/lucas"):
        output_dir = f"/data/lucas/{output_dir}"

    env_vars = {
        "WANDB_PROJECT": f"knowledge-modules-{config_id}",
        "WANDB_MODE": "online",
    }
    os.environ.update(env_vars)

    loop = asyncio.get_running_loop()
    # Register signal handlers for graceful shutdown
    loop.add_signal_handler(signal.SIGTERM, handle_exit_signal, signal.SIGTERM, None)
    loop.add_signal_handler(signal.SIGINT, handle_exit_signal, signal.SIGINT, None)

    use_blob_storage = True  # not os.path.exists("/data/lucas")
    lock_file = f"mttl4879355322/{config_id}/dispatcher.lock"

    # create a folder for this on the blob
    if use_blob_storage:
        blob_engine = BlobStorageEngine()
        blob_engine.create_repo(f"mttl4879355322/{config_id}", exist_ok=True)
        print("created blob")

    job_queue = JobQueue(job_file, lock_file, n_retries, use_blob_storage)
    await job_queue.initialize_queue_file()

    gpu_usage = {i: 0 for i in range(num_gpus)}
    running_tasks = []

    while True:
        # If we received an exit signal (SIGTERM or SIGINT), shut down gracefully
        if got_exit_signal:
            print("Received exit signal, initiating graceful shutdown...")
            # Cancel all running tasks and mark as crashed
            for t, gpu_id, doc_id in running_tasks:
                t.cancel()
            # Wait for all tasks to be cancelled
            done, pending = await asyncio.wait(
                [t for t, _, _ in running_tasks], return_when=asyncio.ALL_COMPLETED
            )
            for completed_task, gpu_id, doc_id in running_tasks:
                # Since these tasks are cancelled, mark them as crashed
                await job_queue.update_job_status(doc_id, "crashed")

            running_tasks.clear()
            break

        # Assign new jobs to free GPUs
        for gpu_id in range(num_gpus):
            while gpu_usage[gpu_id] < jobs_per_gpu:
                doc_id = await job_queue.get_next_job()
                if not doc_id:
                    break
                task = None
                task = asyncio.create_task(
                    run_job(
                        gpu_id, doc_id, config_file, output_dir, train_file, config_id
                    )
                )
                running_tasks.append((task, gpu_id, doc_id))
                gpu_usage[gpu_id] += 1

        if not running_tasks:
            print("No more jobs to process. Exiting.")
            break

        done, pending = await asyncio.wait(
            [task for task, _, _ in running_tasks], return_when=asyncio.FIRST_COMPLETED
        )

        for completed_task in done:
            # Find which task completed
            for idx, (task, gpu_id, doc_id) in enumerate(running_tasks):
                if task == completed_task:
                    gpu_usage[gpu_id] -= 1
                    running_tasks.pop(idx)
                    result_doc_id, return_code = completed_task.result()
                    # Update job status based on return code
                    if return_code == 0:
                        new_status = "done"
                    elif return_code < 0:
                        new_status = "crashed"
                    else:
                        new_status = "failed"
                    await job_queue.update_job_status(result_doc_id, new_status)
                    break
        print_gpu_status(gpu_usage)


if __name__ == "__main__":
    print("starting job launcher")
    asyncio.run(main())
