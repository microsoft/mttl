import asyncio
import json
import os
import random
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from time import sleep

from filelock import AsyncFileLock

from mttl.utils import logger

got_sigterm = False
got_exit_signal = False


def handle_exit_signal(signum, frame):
    logger.warning(
        f"`local_job_launcher.py` received exit signal {signum}, initiating graceful shutdown..."
    )
    global got_exit_signal
    got_exit_signal = True


class JobQueue:
    def __init__(self, job_file, lock_file, n_retries):
        self.job_file = job_file
        self.lock_file = lock_file
        self.queue_file = lock_file.replace(".lock", "_queue.json")

        self.n_retries = n_retries
        print(f"queue file : {self.queue_file}")

        self.launcher_id = f"{random.randint(1000, 9999)}-{os.getpid()}"
        self.heartbeat_file = os.path.join(
            os.path.dirname(lock_file), f"launcher_{self.launcher_id}.txt"
        )
        with open(self.heartbeat_file, "w") as f:
            f.write("heartbeat")

    async def keep_heartbeat_alive(self):
        while True:
            with open(self.heartbeat_file, "a") as f:
                f.write(f"{time.time()}\n")
            await asyncio.sleep(30)

    async def mark_launcher_as_crashed(self, dead_id):
        async with AsyncFileLock(self.lock_file):
            if os.path.exists(self.queue_file):
                print(f"marking launcher {dead_id} as crashed")
                with open(self.queue_file, "r") as f:
                    tasks = json.load(f)
                for task in tasks:
                    if (
                        task.get("launcher_id") == dead_id
                        and task["status"] == "running"
                    ):
                        task["status"] = "crashed"
                        task["launcher_id"] = None
                with open(self.queue_file, "w") as f:
                    json.dump(tasks, f, indent=2)

    async def initialize_queue_file(self):
        async with AsyncFileLock(self.lock_file):
            if not os.path.exists(self.queue_file):
                # copy the original job file to the queue file
                subprocess.run(["cp", self.job_file, self.queue_file])

                # convert the queue_file to the desired format
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

                # Convert to list of {"doc_id": ..., "status": "queued", "retries": 0, "launcher_id": None}
                tasks = [
                    {
                        "doc_id": doc_id,
                        "status": "queued",
                        "retries": 0,
                        "launcher_id": None,
                    }
                    for doc_id in doc_ids
                ]

                with open(self.queue_file, "w") as f:
                    json.dump(tasks, f, indent=2)

    async def get_next_job(self):
        async with AsyncFileLock(self.lock_file):

            with open(self.queue_file, "r") as f:
                tasks = json.load(f)

            doc_id = None
            # Find the first job that is still queued
            for task in tasks:
                if task["status"] == "queued":
                    # Mark as running
                    task["status"] = "running"
                    task["launcher_id"] = self.launcher_id
                    doc_id = task["doc_id"]
                    break

            if doc_id is None:
                # Now, let's see if are some crashed of failed jobs we can retry
                for task in tasks:
                    if task["status"] == "crashed":
                        if task["retries"] < self.n_retries:
                            # Mark as running
                            task["status"] = "running"
                            task["launcher_id"] = self.launcher_id
                            task["retries"] += 1
                            doc_id = task["doc_id"]
                            break

            if doc_id is not None:
                with open(self.queue_file, "w") as f:
                    json.dump(tasks, f, indent=2)

            return doc_id

    async def update_job_status(self, doc_id, new_status):

        if not isinstance(doc_id, list):
            doc_id = [doc_id]

        async with AsyncFileLock(self.lock_file):
            if not os.path.exists(self.queue_file):
                return

            with open(self.queue_file, "r") as f:
                tasks = json.load(f)

            # Update the status of the job with the given doc_id
            for task in tasks:
                if task["doc_id"] in doc_id:
                    # For other statuses, set the status directly
                    task["status"] = new_status
                    if new_status != "running":
                        task["launcher_id"] = None

                    if len(doc_id) == 1:
                        break

            with open(self.queue_file, "w") as f:
                json.dump(tasks, f, indent=2)

    def _print_progress(self, tasks):
        total = len(tasks)
        done = len([task for task in tasks if task["status"] == "done"])
        running = len([task for task in tasks if task["status"] == "running"])
        queued = len([task for task in tasks if task["status"] == "queued"])
        failed = len([task for task in tasks if task["status"] == "failed"])
        crashed = len([task for task in tasks if task["status"] == "crashed"])
        basedir = os.path.dirname(self.lock_file)
        active_launchers_count = sum(
            1
            for f in os.listdir(basedir)
            if f.startswith("launcher_") and f.endswith(".txt")
        )
        print(
            f"***\t\t [{self.launcher_id}, {active_launchers_count}] Progress: {done}/{total} done, {running} running, {queued} queued, {failed} failed, {crashed} crashed\t\t***"
        )

    async def read_queue_file(self):
        async with AsyncFileLock(self.lock_file):
            with open(self.queue_file, "r") as f:
                tasks = json.load(f)
        return tasks

    async def print_progress(self):
        while True:
            tasks = await self.read_queue_file()
            self._print_progress(tasks)
            await asyncio.sleep(5)


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
        job_name = job_name.ljust(9)
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
        "--n-retries",
        type=int,
        default=2,
        help="Number of times to retry a crashed job",
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


async def cleanup_dead_launchers(lock_file, job_queue):
    basedir = os.path.dirname(lock_file)
    while True:
        for fn in os.listdir(basedir):
            if fn.startswith("launcher_") and fn.endswith(".txt"):
                path = os.path.join(basedir, fn)
                st = os.stat(path)
                last_touched = st.st_mtime
                current_time = time.time()
                if (current_time - last_touched) > 600:
                    dead_id = fn[len("launcher_") : -4]
                    print(
                        f"Cleaning up heartbeat file {fn} for launcher {dead_id}. Last touched {last_touched} & current time {current_time}"
                    )
                    if dead_id == job_queue.launcher_id:
                        print(f"Current launcher {dead_id} is dead !!??")
                        continue

                    await job_queue.mark_launcher_as_crashed(dead_id)
                    new_name = f"trace_{dead_id}__{job_queue.launcher_id}.txt"
                    os.rename(path, os.path.join(basedir, new_name))
                    print(f"Renamed {fn} to {new_name}")

        await asyncio.sleep(60)


async def main():
    num_gpus, jobs_per_gpu, job_file, config_id, train_file, n_retries = (
        parse_arguments()
    )
    config_file = f"configs/{config_id}.json"
    output_dir = f"/mnt/output/kms/{config_id}"

    # if running locally, change the output_dir
    if os.path.exists("/data/lucas"):
        output_dir = f"/data/lucas/{output_dir}"

    os.makedirs(output_dir, exist_ok=True)

    env_vars = {
        "WANDB_PROJECT": f"knowledge-modules-{config_id}",
        "WANDB_MODE": "online",
    }
    os.environ.update(env_vars)

    loop = asyncio.get_running_loop()
    # Register signal handlers for graceful shutdown
    loop.add_signal_handler(signal.SIGTERM, handle_exit_signal, signal.SIGTERM, None)
    loop.add_signal_handler(signal.SIGINT, handle_exit_signal, signal.SIGINT, None)

    lock_file = os.path.join(output_dir, "dispatcher.lock")

    job_queue = JobQueue(job_file, lock_file, n_retries)
    await job_queue.initialize_queue_file()

    gpu_usage = {i: 0 for i in range(num_gpus)}
    running_tasks = []

    monitor_task = asyncio.create_task(job_queue.print_progress())
    hearbeat_task = asyncio.create_task(job_queue.keep_heartbeat_alive())
    cleanup_task = asyncio.create_task(cleanup_dead_launchers(lock_file, job_queue))

    while True:
        # If we received an exit signal (SIGTERM or SIGINT), shut down gracefully
        if got_exit_signal:
            logger.warning("Received exit signal, initiating graceful shutdown...")

            running_doc_ids = [doc_id for _, _, doc_id in running_tasks]
            await job_queue.update_job_status(running_doc_ids, "crashed")

            logger.warning("All running tasks marked as crashed. Cancelling jobs.")

            # Cancel all running tasks and mark as crashed
            for t, gpu_id, doc_id in running_tasks:
                t.cancel()

            monitor_task.cancel()
            hearbeat_task.cancel()
            cleanup_task.cancel()

            # print one more time job status
            tasks = await job_queue.read_queue_file()
            job_queue._print_progress(tasks)

            if running_tasks:
                # Wait for all tasks to be cancelled
                done, pending = await asyncio.wait(
                    [t for t, _, _ in running_tasks], return_when=asyncio.ALL_COMPLETED
                )
                running_tasks.clear()

            # delete the heartbeat file
            # os.remove(job_queue.heartbeat_file)
            logger.warning("All running tasks cancelled. Exiting.")
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


if __name__ == "__main__":
    print("starting job launcher")
    asyncio.run(main())
