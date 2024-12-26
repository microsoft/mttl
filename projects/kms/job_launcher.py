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
from mttl.utils import logger

got_sigterm = False
got_exit_signal = False

# Async functions we need to handle
# 1) check if a blob exists
# 2) download a blob
# 3) upload a blob
# Let's first define them as synchronous functions


def handle_exit_signal(signum, frame):
    print(f"Received exit signal {signum}, initiating graceful shutdown...")
    global got_exit_signal
    got_exit_signal = True


class JobQueue:
    def __init__(self, job_file, queue_file, n_retries):
        self.job_file = job_file
        self.queue_file = queue_file
        self.counter = 0

        self.n_retries = n_retries
        self.blob_engine = BlobStorageEngine()

        self.storage_uri, self.container = (
            self.blob_engine._parse_repo_id_to_storage_info(self.queue_file)
        )
        self.queue_name = self.queue_file.split("/")[-1]
        self.blob_service_client = self.blob_engine._get_blob_client(
            self.queue_file, use_async=True
        )
        self.blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=self.queue_name
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container
        )

        print(f"queue file : {self.queue_file}")

    def queue_exists(self):
        return self.blob_engine._blob_exists(self.queue_file)

    async def download_queue(self, lease=None):

        local_filename = self.blob_engine._get_local_filepath(
            self.queue_file, self.queue_name
        )

        async with AsyncFileLock(str(local_filename) + ".lock"):
            with open(file=local_filename, mode="wb") as blob_file:
                download_stream = await self.blob_client.download_blob(lease=lease)
                data = await download_stream.readall()
                # Need an actual lock on this

                blob_file.write(data)

            out = json.load(open(local_filename, "r"))

        return out

        """
        path = await self.blob_engine._async_download_blob(
            self.queue_file, self.queue_file.split("/")[-1], lease=lease, use_cache=False
        )
        return json.load(open(path, 'r'))
        """

    async def upload_queue(self, python_dict, lease=None):
        filename = self.queue_file.split("/")[-1]
        buffer = json.dumps(python_dict, indent=2).encode("utf-8")
        await self.blob_engine._async_upload_blob(
            repo_id=self.queue_file.replace("queue", "QUEUE"),
            filename=filename,
            buffer=buffer,
            lease=lease,
        )

        filename = self.queue_file.replace("queue", f"QUEUE{self.counter}").split("/")[
            -1
        ]
        self.counter += 1
        await self.blob_engine._async_upload_blob(
            repo_id=self.queue_file.replace("queue", "QUEUE"),
            filename=filename,
            buffer=buffer,
        )

    async def initialize_queue_file(self):
        # This function ensures the queue file is in the correct format.

        if not self.queue_exists():
            # copy the original job file to the queue file
            # convert the queue_file to the desired format
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

            await self.upload_queue(tasks)

    async def get_next_job(self):
        if True:
            lease = await self.get_lease()
            if not self.queue_exists():
                raise FileNotFoundError(f"Queue file {self.queue_file} not found.")

            tasks = await self.download_queue(lease=lease)
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

            # upload tasks
            await self.upload_queue(tasks, lease=lease)

            await lease.release()
            print("DOC ID : ", doc_id)
            return doc_id

    async def update_job_status(self, doc_id, new_status):
        if True:
            lease = await self.get_lease()
            assert lease is not None
            if not self.blob_engine._blob_exists(self.queue_file):
                return

            tasks = await self.download_queue(lease=lease)

            self._print_progress(tasks)
            # Update the status of the job with the given doc_id
            for task in tasks:
                if task["doc_id"] == doc_id:
                    print(f"Updating status of {doc_id} to {new_status}")
                    task["status"] = new_status
                    break

            # upload
            await self.upload_queue(tasks, lease=lease)

            await lease.release()

    def _print_progress(self, tasks):
        total = len(tasks)
        done = len([task for task in tasks if task["status"] == "done"])
        running = len([task for task in tasks if task["status"] == "running"])
        queued = len([task for task in tasks if task["status"] == "queued"])
        failed = len([task for task in tasks if task["status"] == "failed"])
        crashed = len([task for task in tasks if task["status"] == "crashed"])
        print(
            f"***\t\tProgress: {done}/{total} done, {running} running, {queued} queued, {failed} failed, {crashed} crashed\t\t***"
        )

    async def print_progress(self):
        while True:
            tasks = await self.download_queue()
            self._print_progress(tasks)
            await asyncio.sleep(30)
            # sleep(10)

    async def get_lease(self, n_retries=15):
        """
        Acquire a lease on the referenced blob before operating.
        This mimics the azure sample's acquire_lease_on_blob_async approach.
        """
        lease_client = None
        n_retried = 0
        should_continue = True
        while should_continue:
            try:
                # Acquire a lease on the blob
                lease_client = await self.blob_client.acquire_lease(lease_duration=-1)
                print("Acquired Lease")
                return lease_client
            except Exception as e:
                if "There is already a lease present." in str(e):
                    print(f"Lease already present. {n_retried} / {n_retries}")
                else:
                    print("Unkown error : ", e)

                n_retried += 1
                should_continue = n_retried < n_retries
                await asyncio.sleep(30)

        raise Exception(
            f"Failed to acquire lease on {self.queue_file} after {n_retries} retries."
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
    # TODO: Remove this
    # command = ['python', 'dummy_train.py']
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
    queue_file = f"mttl4879355322/{config_id}/task_queue.json"

    # create a folder for this on the blob
    blob_engine = BlobStorageEngine()
    blob_engine.create_repo(f"mttl4879355322/{config_id}", exist_ok=True)
    print("created blob")

    job_queue = JobQueue(job_file, queue_file, n_retries)
    await job_queue.initialize_queue_file()

    gpu_usage = {i: 0 for i in range(num_gpus)}
    running_tasks = []

    monitor_task = asyncio.create_task(job_queue.print_progress())

    while True:
        # If we received an exit signal (SIGTERM or SIGINT), shut down gracefully
        if got_exit_signal:
            print("Received exit signal, initiating graceful shutdown...")
            # Cancel all running tasks and mark as crashed
            for t, gpu_id, doc_id in running_tasks + [monitor_task]:
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
                task = asyncio.create_task(
                    run_job(
                        gpu_id, doc_id, config_file, output_dir, train_file, config_id
                    )
                )
                running_tasks.append((task, gpu_id, doc_id))
                gpu_usage[gpu_id] += 1

        if not running_tasks:
            monitor_task.cancel()
            print("No more jobs to process. Exiting.")
            # print progress one last time
            tasks = await job_queue.download_queue()
            job_queue._print_progress(tasks)
            break

        asyncio.create_task(job_queue.print_progress())

        logger.info(f"waiting for {len(running_tasks)} tasks to complete")
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


if __name__ == "__main__":

    print("starting job launcher")
    asyncio.run(main())
