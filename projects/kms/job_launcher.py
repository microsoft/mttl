import asyncio
import json
import os
import subprocess
import sys

from filelock import AsyncFileLock


class JobQueue:
    def __init__(self, job_file, lock_file):
        self.job_file = job_file
        self.lock_file = lock_file
        self.queue_file = os.path.splitext(self.lock_file)[0] + "_queue.json"
        print(f"queue file : {self.queue_file}")

    async def get_next_job(self):
        async with AsyncFileLock(self.lock_file):
            if not os.path.exists(self.queue_file):
                # copy the lock file to the queue file
                subprocess.run(["cp", self.job_file, self.queue_file])

            if not os.path.exists(self.job_file):
                return None

            with open(self.queue_file, "r") as f:
                data = json.load(f)

            # If data is a dict with train/valid/test keys, flatten it.
            if isinstance(data, dict):
                tasks = (
                    data.get("train", []) + data.get("valid", []) + data.get("test", [])
                )
            else:
                # If data is already a single list (previously flattened), just use it.
                tasks = data

            if not tasks:
                return None

            next_job = tasks[0]
            tasks = tasks[1:]  # Remove the retrieved job

            # Write back the updated list of tasks (now always just a list).
            with open(self.queue_file, "w") as f:
                json.dump(tasks, f, indent=2)

            return next_job


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

    # Asynchronously read stdout
    async def stream_output(stream, prefix, job_name):
        job_name = str(job_name)[:10]
        job_name = job_name.ljust(10)
        async for line in stream:
            print(f"{prefix} <{job_name}>: {line.decode().strip()}")

    process = await asyncio.create_subprocess_exec(
        *command,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Run both stdout and stderr stream readers concurrently
    await asyncio.gather(
        stream_output(process.stdout, "STDOUT", doc_id),
        stream_output(process.stderr, "STDERR", doc_id),
    )

    return_code = await process.wait()
    print(f"Process finished with exit code: {return_code}")

    # process = subprocess.Popen(command, env=env)
    # await process.wait()


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
    args = parser.parse_args()
    return (
        args.num_gpus,
        args.jobs_per_gpu,
        args.tasks_file,
        args.config_id,
        args.train_file,
    )


def print_gpu_status(gpu_usage):
    print("=== GPU Status ===")
    for gpu_id, usage in gpu_usage.items():
        print(f"GPU {gpu_id}: {usage} job(s) running")
    print("==================")


async def main():
    num_gpus, jobs_per_gpu, job_file, config_id, train_file = parse_arguments()
    config_file = f"configs/{config_id}.json"
    output_dir = f"/mnt/output/kms/{config_id}"

    # if running locally, change the output_dir
    if os.path.exists("/data/lucas"):
        # prepend
        output_dir = f"/data/lucas/{output_dir}"

    lock_file = f"{output_dir}/job_queue.lock"
    os.makedirs(output_dir, exist_ok=True)

    env_vars = {
        "WANDB_PROJECT": f"knowledge-modules-{config_id}",
        "WANDB_MODE": "online",
    }
    os.environ.update(env_vars)

    job_queue = JobQueue(job_file, lock_file)

    gpu_usage = {i: 0 for i in range(num_gpus)}
    running_tasks = []

    while True:
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
                running_tasks.append((task, gpu_id))
                gpu_usage[gpu_id] += 1
        if not running_tasks:
            print("No more jobs to process. Exiting.")
            break
        # Wait for any task to finish
        done, pending = await asyncio.wait(
            [task for task, _ in running_tasks], return_when=asyncio.FIRST_COMPLETED
        )
        # Update GPU usage and remove completed tasks
        for completed_task in done:
            for idx, (task, gpu_id) in enumerate(running_tasks):
                if task == completed_task:
                    gpu_usage[gpu_id] -= 1
                    running_tasks.pop(idx)
                    break
        print_gpu_status(gpu_usage)


if __name__ == "__main__":
    asyncio.run(main())
