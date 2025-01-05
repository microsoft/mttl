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
    def __init__(self, job_file, output_dir, n_retries, seconds_to_crashed):
        self.job_file = job_file
        self.n_retries = n_retries
        self.output_dir = output_dir
        self.launcher_id = f"{random.randint(1000, 9999)}-{os.getpid()}"
        self.seconds_to_crashed = seconds_to_crashed

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

        self.tasks = doc_ids

    def get_run_status(self, task):
        task_output_dir = os.path.join(self.output_dir, task)
        if os.path.exists(task_output_dir) and len(os.listdir(task_output_dir)) > 0:
            done_file = os.path.join(task_output_dir, "done.txt")
            if os.path.exists(done_file):
                with open(done_file, "r") as f:
                    lines = f.readlines()
                    return_code = int(lines[-1].strip())
                    if return_code == 0:
                        return "done"
                    else:
                        return "failed"
            else:
                # check all the files in `task_output_dir`, and get a timestamp of the most recent file
                # IF nothing happened in the last 15 minutes, we assume the job crashed
                if time.time() - self.get_timestamp(task) > self.seconds_to_crashed:
                    return "crashed"

                # If the job is still running, we assume it's still in progress
                return "running"
        else:
            return "queued"

    def get_timestamp(self, task):
        task_output_dir = os.path.join(self.output_dir, task)
        files = os.listdir(task_output_dir)

        if len(files) > 0:
            return max(
                os.path.getmtime(os.path.join(task_output_dir, f)) for f in files
            )

    def get_next_job(self):

        # shuffle the tasks
        tasks = self.tasks.copy()
        random.shuffle(tasks)

        doc_id = None
        for task in tasks:
            status = self.get_run_status(task)

            if status == "queued":
                doc_id = task
                break

        if doc_id is None:
            # Let's look for crashed jobs
            for task in tasks:
                status = self.get_run_status(task)
                if status == "crashed":
                    doc_id = task
                    # clean out the task directory
                    task_output_dir = os.path.join(self.output_dir, task)
                    timestamp = self.get_timestamp(task)
                    # convert timestsamp to DD-MM-MM-YYYY HH:MM:SS
                    timestamp_str = time.strftime(
                        "%d-%m-%Y %H:%M:%S", time.localtime(timestamp)
                    )
                    logger.warning(f"Cleaning up crashed job {task}: {timestamp_str}")
                    os.system(f"rm -rf {task_output_dir}")
                    break

        return doc_id

    def _print_progress(self):
        all_status = {"done": 0, "failed": 0, "queued": 0, "crashed": 0, "running": 0}

        for task in self.tasks:
            status = self.get_run_status(task)
            all_status[status] += 1

        # build a single line string will all the statuses
        status_str = " | ".join([f"{k}: {v}" for k, v in all_status.items()])
        print(f"*** [Progress : {self.launcher_id}]   {status_str}     ***")

    async def print_progress(self):
        print("starting print progress task")
        while True:
            self._print_progress()
            await asyncio.sleep(5)


async def run_job(
    gpu_id, doc_id, config_file, output_dir, train_script, config_id, launcher_id
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(env.get("GPU_OFFSET", 0)) + int(gpu_id))
    command = [
        "python",
        f"{train_script}.py",
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

    heartbeat_file = os.path.join(output_dir, doc_id, "heartbeat.txt")

    async def doc_heartbeat(launcher_id):
        while True:
            with open(heartbeat_file, "a") as f:
                f.write(f"{time.time()} - {launcher_id}\n")
            await asyncio.sleep(30)

    # Make sure doc_id directory exists
    os.makedirs(os.path.join(output_dir, doc_id), exist_ok=True)

    heartbeat_task = asyncio.create_task(doc_heartbeat(launcher_id))

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
    finally:
        heartbeat_task.cancel()

    return_code = await process.wait()
    print(f"Process finished with exit code: {return_code} for DOC_ID={doc_id}")

    done_file = os.path.join(output_dir, doc_id, "done.txt")

    with open(done_file, "a") as f:
        f.write(str(return_code))

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
        "--train-script",
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
    parser.add_argument(
        "--seconds-to-crashed",
        type=int,
        default=15 * 60,
    )
    args = parser.parse_args()
    return args


async def main():
    args = parse_arguments()
    config_file = f"configs/{args.config_id}.json"
    output_dir = f"/mnt/output/kms/{args.config_id}"

    # if running locally, change the output_dir
    if os.path.exists("/data/lucas"):
        output_dir = f"/data/lucas/{output_dir}"

    os.makedirs(output_dir, exist_ok=True)
    loop = asyncio.get_running_loop()

    # Register signal handlers for graceful shutdown
    loop.add_signal_handler(signal.SIGTERM, handle_exit_signal, signal.SIGTERM, None)
    loop.add_signal_handler(signal.SIGINT, handle_exit_signal, signal.SIGINT, None)
    job_queue = JobQueue(
        args.tasks_file, output_dir, args.n_retries, args.seconds_to_crashed
    )

    gpu_usage = {i: 0 for i in range(args.num_gpus)}
    running_tasks = []

    monitor_task = asyncio.create_task(job_queue.print_progress())

    while True:
        # If we received an exit signal (SIGTERM or SIGINT), shut down gracefully
        if got_exit_signal:
            logger.warning("Received exit signal, initiating graceful shutdown...")

            running_doc_ids = [doc_id for _, _, doc_id in running_tasks]
            logger.warning("All running tasks marked as crashed. Cancelling jobs.")

            # Cancel all running tasks and mark as crashed
            for t, gpu_id, doc_id in running_tasks:
                t.cancel()

            monitor_task.cancel()
            job_queue._print_progress()

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
        for gpu_id in range(args.num_gpus):
            while gpu_usage[gpu_id] < args.jobs_per_gpu:
                doc_id = job_queue.get_next_job()
                if not doc_id:
                    break
                task = asyncio.create_task(
                    run_job(
                        gpu_id,
                        doc_id,
                        config_file,
                        output_dir,
                        args.train_script,
                        args.config_id,
                        job_queue.launcher_id,
                    )
                )
                running_tasks.append((task, gpu_id, doc_id))
                gpu_usage[gpu_id] += 1

        if not running_tasks:
            monitor_task.cancel()
            print("No more jobs to process. Exiting.")
            job_queue._print_progress()
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


if __name__ == "__main__":
    print("starting job launcher")
    asyncio.run(main())
